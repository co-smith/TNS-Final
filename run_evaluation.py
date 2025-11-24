import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from policy_proposal_labeler import DisinformationLabeler

def run_test():    
    labeler = DisinformationLabeler()
    
    # Load the Test Set
    df = pd.read_csv('data/test_data.csv')
    
    # Clean up NaNs
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    preds = []
    truth = df['label'].tolist()
        
    start = time.time()
    
    for index, row in df.iterrows():
        # Ask the labeler for a decision
        labels, score = labeler.moderate_post(row)
        
        # Convert label to 0 or 1
        if 'suspected-russian-disinfo' in labels:
            preds.append(1)
        else:
            preds.append(0)
            
    end = time.time()
    
    # Calculate stats
    acc = accuracy_score(truth, preds)
    prec = precision_score(truth, preds, zero_division=0)
    rec = recall_score(truth, preds, zero_division=0)
    
    print("-" * 30)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"Total Time: {end - start:.2f}s")
    print("-" * 30)
    
    # Save Confusion Matrix data for the graph script
    cm = confusion_matrix(truth, preds)
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    np.save('graphs/confusion_matrix_data.npy', cm)
    print("Saved confusion matrix data.")

if __name__ == "__main__":
    run_test()