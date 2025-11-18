import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
# Import the class so we can access class variables
from policy_proposal_labeler import DisinformationLabeler 

def tune():
    print("Loading Data & Model")
    labeler = DisinformationLabeler()
    
    # Load the 30% Tuning Set
    df = pd.read_csv('data/tuning_data.csv')
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    ground_truth = df['label'].tolist()

    print(f"Pre-Calculating Scores for {len(df)} examples (Using Cross-Encoder)...")
    results = []
    
    for index, row in df.iterrows():
        uri_handle = labeler._extract_handle(row.get('clean_uri', ''))
        text = row.get('translated_text', row.get('text', ''))
        
        s_score = labeler._check_source(uri_handle) 
        c_score = labeler._check_content(text)
        results.append((s_score, c_score))

    print("Starting Grid Search")
    best_f_half = 0 # Now maximizing F0.5
    best_params = (0, 0)
    
    # Tuning ranges remain the same
    c_main_thresholds = np.linspace(-2.0, 2.0, 10).round(2)
    s_hybrid_thresholds = [0.0, 0.5, 0.7]
    
    # FIX: Access fixed threshold directly from the class
    S_HYBRID_C_THRESHOLD = DisinformationLabeler.CONTENT_HYBRID_THRESHOLD
    
    print(f"Testing {len(s_hybrid_thresholds) * len(c_main_thresholds)} combinations...")

    for s_hybrid in s_hybrid_thresholds:
        for c_main in c_main_thresholds:
            predictions = []
            for (s_score, c_score) in results:
                
                # APPLYING THE MODERATE_POST LOGIC:
                
                if s_score == 1.0:
                    pred = 1
                    
                elif s_score > s_hybrid and c_score > S_HYBRID_C_THRESHOLD: 
                    pred = 1
                    
                elif c_score > c_main:
                    pred = 1
                    
                else:
                    pred = 0
                    
                predictions.append(pred)
            
            # Calculate F0.5 Score (Prioritizes Precision over Recall)
            f_half = f1_score(ground_truth, predictions, beta=0.5, zero_division=0)
            
            if f_half > best_f_half:
                best_f_half = f_half
                best_params = (s_hybrid, c_main)
                best_p = precision_score(ground_truth, predictions, zero_division=0)
                best_r = recall_score(ground_truth, predictions, zero_division=0)

    print("\n" + "="*30)
    print("      OPTIMIZATION RESULTS (F0.5 Focused)")
    print("="*30)
    print(f"Best F0.5 Score: {best_f_half:.2%} (on Tuning Set)")
    print(f"Precision:     {best_p:.2%}")
    print(f"Recall:        {best_r:.2%}")
    print("-" * 20)
    print(f"Optimal SOURCE_HYBRID_THRESHOLD:  {best_params[0]}")
    print(f"Optimal CONTENT_MAIN_THRESHOLD:   {best_params[1]}")

if __name__ == "__main__":
    tune()