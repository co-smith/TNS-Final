import pandas as pd
import time
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from policy_proposal_labeler import DisinformationLabeler

def run_test():
    print("Starting Evaluation")
    if not os.path.exists('data/test_data.csv'):
         print("Error: data/test_data.csv missing.")
         return

    labeler = DisinformationLabeler()
    
    df = pd.read_csv('data/test_data.csv')
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    predictions = []
    ground_truth = df['label'].tolist()
    processing_times = []

    for index, row in df.iterrows():
        start_time = time.time()
        labels, score = labeler.moderate_post(row)
        end_time = time.time()
        
        processing_times.append(end_time - start_time)
        is_flagged = 1 if 'disinfo-watch' in labels else 0
        predictions.append(is_flagged)
        
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    avg_time = sum(processing_times) / len(processing_times)

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"Avg Latency: {avg_time*1000:.1f} ms")
    
    cm = confusion_matrix(ground_truth, predictions)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    
    # SAVE RESULTS FOR GRAPH.PY
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    np.save('graphs/confusion_matrix_data.npy', cm)
    print("Confusion matrix data saved to graphs/confusion_matrix_data.npy")

if __name__ == "__main__":
    run_test()