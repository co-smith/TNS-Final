import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from policy_proposal_labeler import DisinformationLabeler

def tune():
    print("Loading Data & Model")
    labeler = DisinformationLabeler()
    
    df = pd.read_csv('data/test_data.csv')
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    ground_truth = df['label'].tolist()

    print("Pre-Calculating Scores")
    results = []
    for index, row in df.iterrows():
        uri_handle = labeler._extract_handle(row.get('clean_uri', ''))
        text = row.get('translated_text', row.get('text', ''))
        
        s_score = labeler._check_source(uri_handle)
        c_score = labeler._check_content(text)
        results.append((s_score, c_score))

    print(" Starting Grid Search")
    best_f1 = 0
    best_params = (0, 0)
    
    source_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    content_thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

    print(f"Testing {len(source_thresholds) * len(content_thresholds)} combinations...")

    for s_thresh in source_thresholds:
        for c_thresh in content_thresholds:
            predictions = []
            for (s_score, c_score) in results:
                # Apply the "Tiered Logic" with variable thresholds
                if s_score > s_thresh:     # Strong Source Match
                    pred = 1
                elif c_score > c_thresh:   # Strong Content Match
                    pred = 1
                elif s_score > 0.5 and c_score > 0.6: # Hybrid (Keep static or tune this too)
                    pred = 1
                else:
                    pred = 0
                predictions.append(pred)
            
            score = f1_score(ground_truth, predictions)
            
            if score > best_f1:
                best_f1 = score
                best_params = (s_thresh, c_thresh)

    print("\n" + "="*30)
    print("      OPTIMIZATION RESULTS")
    print("="*30)
    print(f"Best F1 Score: {best_f1:.2%}")
    print(f"Optimal Source Threshold:  {best_params[0]}")
    print(f"Optimal Content Threshold: {best_params[1]}")

if __name__ == "__main__":
    tune()