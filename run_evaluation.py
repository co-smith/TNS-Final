import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from policy_proposal_labeler import DisinformationLabeler

def run_test():
    print("Starting Evaluation")
    labeler = DisinformationLabeler()
    
    # Using test data (70% split)
    df = pd.read_csv('data/test_data.csv')
    df['clean_uri'] = df['clean_uri'].fillna('')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    predictions = []
    ground_truth = df['label'].tolist()
    processing_times = []

    print(f"Processing {len(df)} posts...")

    for index, row in df.iterrows():
        start_time = time.time()
        labels, score = labeler.moderate_post(row)
        end_time = time.time()
        
        processing_times.append(end_time - start_time)
        is_flagged = 1 if 'disinfo-watch' in labels else 0
        predictions.append(is_flagged)
        
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    avg_time = sum(processing_times) / len(processing_times)

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"Avg Latency: {avg_time*1000:.1f} ms")
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

if __name__ == "__main__":
    run_test()