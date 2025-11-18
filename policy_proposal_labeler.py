import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class DisinformationLabeler:
    # --- THRESHOLDS (Fixed for Stable Precision) ---
    SOURCE_HIGH_CONFIDENCE = 0.9  
    # FIX: Setting the content threshold to a conservative positive value guarantees TN > 0.
    CONTENT_MAIN_THRESHOLD = 0.5  # Start at 0.5. Requires positive Cross-Encoder evidence.
    
    # FIX: Declared unused variables to prevent 'AttributeError' in tune_thresholds.py
    SOURCE_HYBRID_THRESHOLD = 0.5 
    CONTENT_HYBRID_THRESHOLD = 0.0
    
    def __init__(self, narrative_file="data/known_narratives.csv", blacklist_file="data/sus_tele.csv"):
        print("Initializing Labeler with Retrieve & Re-Rank...")
        
        # Bi-Encoder for fast retrieval
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Cross-Encoder for accurate verification
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        narrative_df = pd.read_csv(narrative_file)
        self.known_narratives = narrative_df['narrative'].dropna().tolist()
        self.narrative_embeddings = self.bi_encoder.encode(self.known_narratives)
        print(f"Loaded {len(self.known_narratives)} known narratives.")

        blacklist_df = pd.read_csv(blacklist_file)
        self.blacklist = blacklist_df['tele_handle'].dropna().tolist()
        print(f"Loaded {len(self.blacklist)} blacklisted channels.")

    def _extract_handle(self, clean_uri):
        if not isinstance(clean_uri, str):
            return ""
        return clean_uri.split('/')[0]

    def _check_source(self, handle):
        if not handle or not self.blacklist:
            return 0.0
        
        match = process.extractOne(handle, self.blacklist, scorer=fuzz.WRatio)
        if match and match[1] > self.SOURCE_HIGH_CONFIDENCE:
            return 1.0 
        return 0.0

    def _check_content(self, text):
        if not text or not self.known_narratives:
            return 0.0
        
        # 1. RETRIEVE (Bi-Encoder)
        text_embedding = self.bi_encoder.encode([text])
        similarities = cosine_similarity(text_embedding, self.narrative_embeddings)[0]
        
        top_k_indices = np.argsort(similarities)[-5:]
        top_candidates = [self.known_narratives[i] for i in top_k_indices]
        
        # 2. RE-RANK (Cross-Encoder)
        pairs = [[text, candidate] for candidate in top_candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        return float(np.max(cross_scores))

    def moderate_post(self, post_row):
        uri_handle = self._extract_handle(post_row.get('clean_uri', ''))
        text = post_row.get('translated_text', post_row.get('text', ''))
        
        source_score = self._check_source(uri_handle)
        content_score = self._check_content(text)
        
        labels = []
        final_confidence = 0.0
        
        # 1. High Source Match
        if source_score == 1.0:
            final_confidence = 1.0
            labels.append('disinfo-watch')
            
        # 2. Content Match (Must be greater than a positive, conservative threshold)
        elif content_score > self.CONTENT_MAIN_THRESHOLD:
            final_confidence = content_score
            labels.append('disinfo-watch')
            
        return labels, final_confidence