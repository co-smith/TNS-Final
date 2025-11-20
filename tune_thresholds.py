import pandas as pd
import numpy as np
import json
from sklearn.metrics import fbeta_score, precision_score, recall_score
from policy_proposal_labeler import DisinformationLabeler 

def tune():
    print("Initializing Labeler for tuning...")
    labeler = DisinformationLabeler()
    
    # Load the Tuning set
    df = pd.read_csv('data/tuning_data.csv')
    
    # Fill NaNs just like we do in production
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    ground_truth = df['label'].tolist()
    scores = []
    
    print(f"Calculating scores for {len(df)} rows...")
    
    # Run the EXACT same logic as the class
    for index, row in df.iterrows():
        s = labeler.calculate_score(row)
        scores.append(s)

    # Search for the best threshold
    # We check 0.1, 0.11, 0.12 ... up to 0.9
    best_f = -1
    best_thresh = 0.0
    
    thresholds = np.linspace(0.1, 0.9, 81) 
    
    for t in thresholds:
        # If score > t, predict 1 (Disinfo), else 0 (Safe)
        preds = [1 if s > t else 0 for s in scores]
        
        # We use F0.5 because we care more about Precision (not banning safe people)
        f_score = fbeta_score(ground_truth, preds, beta=0.5, zero_division=0)
        
        if f_score > best_f:
            best_f = f_score
            best_thresh = t

    print(f"Best Threshold Found: {best_thresh:.3f}")
    print(f"Best F0.5 Score: {best_f:.3f}")
    
    # Save to config
    new_config = {
        "weights": labeler.weights,
        "decision_threshold": float(round(best_thresh, 3))
    }
    
    with open('config.json', 'w') as f:
        json.dump(new_config, f, indent=4)
    
    print("Saved new settings to config.json")

if __name__ == "__main__":
    tune()