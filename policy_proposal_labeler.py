import pandas as pd
import numpy as np
import re
import os
import json
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class DisinformationLabeler:
    def __init__(self):
        # Load settings from config file so we don't hardcode magic numbers
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.weights = config['weights']
            self.threshold = config['decision_threshold']
        
        # Load the AI models (MiniLM is fast and good enough)
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Load our Knowledge Base (narratives we know are bad)
        narrative_df = pd.read_csv("data/known_narratives.csv")
        self.known_narratives = narrative_df['narrative'].dropna().tolist()

        # Pre-encode the knowledge base for fast searching
        if self.known_narratives:
            self.narrative_embeddings = self.bi_encoder.encode(self.known_narratives)
        else:
            self.narrative_embeddings = None

        # Load the Telegram Blacklist
        blacklist_df = pd.read_csv("data/sus_tele.csv")
        self.blacklist = blacklist_df['tele_handle'].dropna().tolist()
        self.blacklist_set = set(x.lower() for x in self.blacklist)

    def _extract_handle(self, url):
        # cleans up a URL to get just the channel name
        if not isinstance(url, str): return ""
        if "://" in url: url = url.split("://")[1]
        if "t.me/" in url: url = url.split("t.me/")[1]
        return url.split('/')[0].lower()

    def _check_source(self, handle):
        # Checks if the handle is in our bad list
        if not handle: return 0.0
        handle = handle.lower()
        
        # Direct match
        if handle in self.blacklist_set: return 1.0
        
        # Fuzzy match (in case of slight typos)
        match = process.extractOne(handle, self.blacklist, scorer=fuzz.ratio)
        if match and match[1] > 80:
            return match[1] / 100.0
        return 0.0

    def _check_content(self, text):
        # Checks semantic similarity to known disinfo
        if not isinstance(text, str) or len(text) < 5: return 0.0
        
        # Bi-Econder
        text_embedding = self.bi_encoder.encode([text])
        sims = cosine_similarity(text_embedding, self.narrative_embeddings)[0]
        
        # Get top 5 closest matches
        top_indices = np.argsort(sims)[-5:]
        candidates = [self.known_narratives[i] for i in top_indices]
        
        # Cross-Encoder
        pairs = [[text, cand] for cand in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Take the highest score found
        max_logit = float(np.max(scores))
        
        # Convert logit to probability using sigmoid
        return 1 / (1 + np.exp(-max_logit))

    def calculate_score(self, row):
        # Main helper to get the final number
        
        # Get handle
        uri = self._extract_handle(row.get('clean_uri', ''))
        text = row.get('translated_text', row.get('text', ''))
        
        # Regex fallback if url column is empty
        if not uri and isinstance(text, str) and "t.me/" in text:
             match = re.search(r't\.me/([a-zA-Z0-9_]+)', text)
             if match: uri = match.group(1).lower()

        # Get the two sub-scores
        source_score = self._check_source(uri)
        content_score = self._check_content(text)
        
        # Weighted average
        final = (self.weights['source'] * source_score) + \
                (self.weights['content'] * content_score)
        
        return final

    def moderate_post(self, row):
        # Returns the label based on the score
        score = self.calculate_score(row)
        labels = []
        
        # Check against the threshold from config
        if score > self.threshold:
            labels.append('disinfo-watch')
            
        return labels, score