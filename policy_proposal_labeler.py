import pandas as pd
import numpy as np
import re
import os
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class DisinformationLabeler:
    # --- CONFIGURATION ---
    # Adjust these weights to prioritize Source (Blacklist) vs Content (Narratives)
# --- CONFIGURATION ---
    # 50/50 split ensures either a strict source OR a strong narrative can trigger it.
    WEIGHT_SOURCE = 0.5
    WEIGHT_CONTENT = 0.5
    
    # Set Threshold slightly below 0.5 so that a perfect match (0.5 contribution) passes.
    DECISION_THRESHOLD = 0.45
    
    def __init__(self, narrative_file="data/known_narratives.csv", blacklist_file="data/sus_tele.csv"):
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Load Knowledge Base
        if os.path.exists(narrative_file):
            narrative_df = pd.read_csv(narrative_file)
            if 'narrative' in narrative_df.columns:
                self.known_narratives = narrative_df['narrative'].dropna().tolist()
            else:
                self.known_narratives = []
        else:
            self.known_narratives = []

        if self.known_narratives:
            self.narrative_embeddings = self.bi_encoder.encode(self.known_narratives)
        else:
            self.narrative_embeddings = None

        # Load Blacklist
        if os.path.exists(blacklist_file):
            blacklist_df = pd.read_csv(blacklist_file)
            self.blacklist = blacklist_df['tele_handle'].dropna().tolist()
            # Create a set for fast exact matching
            self.blacklist_set = set(x.lower() for x in self.blacklist)
        else:
            self.blacklist = []
            self.blacklist_set = set()

    def _extract_handle(self, clean_uri):
        """Extracts potential telegram handle from URI string."""
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
        """
        Returns a confidence score (0.0 to 1.0) for the source match.
        """
        if not handle or not self.blacklist:
            return 0.0
        
        handle_lower = handle.lower()
        
        # 1. Exact Match (100% Confidence)
        if handle_lower in self.blacklist_set:
            return 1.0

        # 2. Fuzzy Match (Partial Confidence)
        # Extract ratio returns 0-100, we normalize to 0.0-1.0
        match = process.extractOne(handle, self.blacklist, scorer=fuzz.ratio)
        if match:
            score = match[1]
            # Only consider fuzzy matches if they are somewhat decent (>80)
            if score > 80:
                return score / 100.0
            
        return 0.0

    def _check_content(self, text):
        """
        Returns the raw logit score for content similarity.
        """
        if not isinstance(text, str) or len(text) < 5 or not self.known_narratives:
            return -10.0
        
        # 1. RETRIEVE (Bi-Encoder)
        text_embedding = self.bi_encoder.encode([text])
        similarities = cosine_similarity(text_embedding, self.narrative_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-5:]
        top_candidates = [self.known_narratives[i] for i in top_k_indices]
        
        # 2. RE-RANK (Cross-Encoder)
        pairs = [[text, candidate] for candidate in top_candidates]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Return the highest score found
        return float(np.max(cross_scores))

    def _sigmoid(self, x):
        """Converts raw logit scores (-inf, inf) to probability (0, 1)"""
        return 1 / (1 + np.exp(-x))

    def moderate_post(self, post_row):
        # 1. Extract Source Info
        uri_handle = self._extract_handle(post_row.get('clean_uri', ''))
        text = post_row.get('translated_text', post_row.get('text', ''))
        
        # Fallback regex extraction for handle
        if not uri_handle and isinstance(text, str) and "t.me/" in text:
             match = re.search(r't\.me/([a-zA-Z0-9_]+)', text)
             if match:
                 uri_handle = match.group(1).lower()

        # 2. Calculate Component Scores
        source_score = self._check_source(uri_handle)    # 0.0 to 1.0
        
        content_logit = self._check_content(text)        # Raw logit (e.g. -5.0 to 5.0)
        content_prob = self._sigmoid(content_logit)      # 0.0 to 1.0
        
        # 3. Weighted Sum
        final_score = (self.WEIGHT_SOURCE * source_score) + (self.WEIGHT_CONTENT * content_prob)
        
        # 4. Decision
        labels = []
        if final_score > self.DECISION_THRESHOLD:
            labels.append('disinfo-watch')
            
        return labels, final_score