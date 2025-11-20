import pandas as pd
import numpy as np
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from policy_proposal_labeler import DisinformationLabeler 

def tune():
    if not os.path.exists('data/tuning_data.csv'):
        print("Error: data/tuning_data.csv not found. Run generate_resources.py first.")
        return

    labeler = DisinformationLabeler()
    
    df = pd.read_csv('data/tuning_data.csv')
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    ground_truth = df['label'].tolist()
    results = []
    
    print(f"Scoring {len(df)} tuning examples against new Knowledge Base...")
    
    for index, row in df.iterrows():
        uri_handle = labeler._extract_handle(row.get('clean_uri', ''))
        text = row.get('translated_text', row.get('text', ''))
        
        s_score = labeler._check_source(uri_handle) 
        c_score = labeler._check_content(text)
        results.append((s_score, c_score))

    # --- DIAGNOSTICS ---
    safe_scores = [r[1] for i, r in enumerate(results) if ground_truth[i] == 0]
    disinfo_scores = [r[1] for i, r in enumerate(results) if ground_truth[i] == 1]

    avg_safe = np.mean(safe_scores) if safe_scores else 0
    avg_disinfo = np.mean(disinfo_scores) if disinfo_scores else 0
    
    print("      SCORE DIAGNOSTICS")
    print(f"Avg SAFE Score:    {avg_safe:.3f}")
    print(f"Avg DISINFO Score: {avg_disinfo:.3f}")
    
    if avg_disinfo < avg_safe:
        print("WARNING: Disinfo scores are lower than Safe scores. Your Knowledge Base may not match your Tuning Data.")
    
    search_start = max(0.0, avg_safe) 
    search_end = max(search_start + 2.0, avg_disinfo + 1.0)
    
    c_main_thresholds = np.linspace(search_start, search_end, 100).round(2)
    s_hybrid_thresholds = [0.0, 0.5, 0.7]
    S_HYBRID_C_THRESHOLD = 1.0 

    best_f_half = -1
    best_params = (0, 0)
    best_metrics = (0, 0, 0, 0)

    for s_hybrid in s_hybrid_thresholds:
        for c_main in c_main_thresholds:
            predictions = []
            for (s_score, c_score) in results:
                if s_score == 1.0:
                    pred = 1
                elif s_score > s_hybrid and c_score > S_HYBRID_C_THRESHOLD: 
                    pred = 1
                elif c_score > c_main:
                    pred = 1
                else:
                    pred = 0
                predictions.append(pred)
            
            f_half = fbeta_score(ground_truth, predictions, beta=0.5, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
            
            if tn > 0 and f_half > best_f_half:
                best_f_half = f_half
                best_params = (s_hybrid, c_main)
                p = precision_score(ground_truth, predictions, zero_division=0)
                r = recall_score(ground_truth, predictions, zero_division=0)
                best_metrics = (p, r, tn, tp)

    print("      OPTIMIZATION RESULTS")
    print(f"Best F0.5 Score: {best_f_half:.2%}")
    print(f"Optimal SOURCE_HYBRID_THRESHOLD:  {best_params[0]}")
    print(f"Optimal CONTENT_MAIN_THRESHOLD:   {best_params[1]}")

if __name__ == "__main__":
    tune()