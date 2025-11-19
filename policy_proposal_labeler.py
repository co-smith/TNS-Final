import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class DisinformationLabeler:
    SOURCE_HIGH_CONFIDENCE = 95  # Strict Exact/Near-Exact Matching
    
    CONTENT_MAIN_THRESHOLD = 0.0  
    
    CONTENT_HYBRID_THRESHOLD = 1.0 
    SOURCE_HYBRID_THRESHOLD = 0.7
    
    def __init__(self, narrative_file="data/known_narratives.csv", blacklist_file="data/sus_tele.csv"):
        
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        narrative_df = pd.read_csv(narrative_file)
        self.known_narratives = narrative_df['narrative'].dropna().tolist()
        self.narrative_embeddings = self.bi_encoder.encode(self.known_narratives)

        blacklist_df = pd.read_csv(blacklist_file)
        self.blacklist = blacklist_df['tele_handle'].dropna().tolist()

    def _extract_handle(self, clean_uri):
        """ Robustly extracts a lowercase handle from various URL formats. """
        if not isinstance(clean_uri, str):
            return ""
        if "://" in clean_uri:
            clean_uri = clean_uri.split("://")[1]
        if "t.me/" in clean_uri:
            clean_uri = clean_uri.split("t.me/")[1]
        elif "telegram.me/" in clean_uri:
            clean_uri = clean_uri.split("telegram.me/")[1]
        return clean_uri.split('/')[0].lower()

    def _check_source(self, handle):
        if not handle or not self.blacklist:
            return 0.0
        
        # STRICT CHECK 1: Exact match (Case-insensitive)
        handle_lower = handle.lower()
        for suspect in self.blacklist:
            if suspect.lower() == handle_lower:
                return 1.0

        # STRICT CHECK 2: High-Confidence Ratio only
        # 'fuzz.Ratio' is stricter than 'WRatio'.
        match = process.extractOne(handle, self.blacklist, scorer=fuzz.ratio)
        if match and match[1] > self.SOURCE_HIGH_CONFIDENCE:
            return 1.0 
            
        return 0.0

    def _check_content(self, text):
        # Guard against empty/bad text
        if not isinstance(text, str) or len(text) < 5 or not self.known_narratives:
            return -10.0
        
        # 1. RETRIEVE
        text_embedding = self.bi_encoder.encode([text])
        similarities = cosine_similarity(text_embedding, self.narrative_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-5:]
        top_candidates = [self.known_narratives[i] for i in top_k_indices]
        
        # 2. RE-RANK
        pairs = [[text, candidate] for candidate in top_candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        return float(np.max(cross_scores))

    def moderate_post(self, post_row):
        uri_handle = self._extract_handle(post_row.get('clean_uri', ''))
        text = post_row.get('translated_text', post_row.get('text', ''))
        
        # Fallback: extract handle from text body
        if not uri_handle and isinstance(text, str) and "t.me/" in text:
             match = re.search(r't\.me/([a-zA-Z0-9_]+)', text)
             if match:
                 uri_handle = match.group(1).lower()

        source_score = self._check_source(uri_handle)
        content_score = self._check_content(text)
        
        labels = []
        final_confidence = 0.0
        
        # 1. Source Match
        if source_score == 1.0:
            final_confidence = 1.0
            labels.append('disinfo-watch')
            
        # 2. Hybrid Match
        elif source_score > self.SOURCE_HYBRID_THRESHOLD and content_score > self.CONTENT_HYBRID_THRESHOLD:
            final_confidence = (source_score + content_score) / 2
            labels.append('disinfo-watch')
            
        # 3. Content Match
        elif content_score > self.CONTENT_MAIN_THRESHOLD:
            final_confidence = content_score
            labels.append('disinfo-watch')
            
        return labels, final_confidence