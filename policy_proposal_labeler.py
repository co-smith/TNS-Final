import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DisinformationLabeler:
    def __init__(self, narrative_file="data/known_narratives.csv", blacklist_file="data/sus_tele.csv"):
        print("Initializing Labeler")
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')        
        narrative_df = pd.read_csv(narrative_file)
        self.known_narratives = narrative_df['narrative'].dropna().tolist()
        # Pre-compute embeddings for speed
        self.narrative_embeddings = self.model.encode(self.known_narratives)
        print(f"Loaded {len(self.known_narratives)} known narratives.")

        blacklist_df = pd.read_csv(blacklist_file)
        self.blacklist = blacklist_df['tele_handle'].dropna().tolist()
        print(f"Loaded {len(self.blacklist)} blacklisted channels.")

    def _extract_handle(self, clean_uri):
        """Extracts 'crimeanwind' from 'crimeanwind/...'"""
        if not isinstance(clean_uri, str):
            return ""
        return clean_uri.split('/')[0]

    def _check_source(self, handle):
        """Branch 1: Metadata/Source Check"""
        if not handle or not self.blacklist:
            return 0.0
        
        # Fuzzy match handle against blacklist
        match = process.extractOne(handle, self.blacklist, scorer=fuzz.WRatio)
        if match:
            score = match[1]
            if score > 85: # Threshold for "This is definitely the same channel"
                return 1.0
            if score > 70:
                return 0.5
        return 0.0

    def _check_content(self, text):
        """Branch 2: Semantic Content Check"""
        if not text or not self.known_narratives:
            return 0.0
        
        text_embedding = self.model.encode([text])
        
        similarities = cosine_similarity(text_embedding, self.narrative_embeddings)
        
        return float(np.max(similarities))

    def moderate_post(self, post_row):
        """
        Final Optimized Logic (F1 Score: 71.7%)
        """
        uri_handle = self._extract_handle(post_row.get('clean_uri', ''))
        text = post_row.get('translated_text', post_row.get('text', ''))
        
        source_score = self._check_source(uri_handle)
        content_score = self._check_content(text)
        
        final_confidence = 0.0
        labels = []
        
        # Condition 1: Source Match (Threshold lowered to 0.5 by optimizer)
        if source_score > 0.5:
            final_confidence = 1.0
            labels.append('disinfo-watch')
            
        # Condition 2: Content Match (Threshold raised to 0.8 by optimizer)
        elif content_score > 0.8:
            final_confidence = content_score
            labels.append('disinfo-watch')
            
        # Condition 3: Hybrid
        elif source_score > 0.4 and content_score > 0.6:
            final_confidence = (source_score + content_score) / 2
            labels.append('disinfo-watch')
            
        return labels, final_confidence