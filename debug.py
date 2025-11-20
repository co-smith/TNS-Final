import pandas as pd
from policy_proposal_labeler import DisinformationLabeler

def debug():
    print("--- DEBUGGING FALSE NEGATIVES ---")
    labeler = DisinformationLabeler()
    
    df = pd.read_csv('data/test_data.csv')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    missed_count = 0
    
    for index, row in df.iterrows():
        # We only care about Disinfo (Label=1) that we missed
        if row['label'] == 1:
            labels, score = labeler.moderate_post(row)
            
            if not labels: # We missed it (False Negative)
                missed_count += 1
                if missed_count <= 5: # Only show first 5
                    print(f"\n[MISSED ITEM #{missed_count}]")
                    text = str(row.get('translated_text', ''))[:100]
                    print(f"Text: '{text}...'")
                    
                    # What did the bot think?
                    content_score = labeler._check_content(row.get('translated_text', ''))
                    print(f"Bot Content Score: {content_score:.4f} (Threshold needed: >2.0)")
                    
                    # What was the closest match?
                    # (Re-running retrieval manually to show you)
                    embedding = labeler.bi_encoder.encode([row.get('translated_text', '')])
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np
                    sims = cosine_similarity(embedding, labeler.narrative_embeddings)[0]
                    best_idx = np.argmax(sims)
                    print(f"Closest Known Narrative: '{labeler.known_narratives[best_idx][:80]}...'")

    print(f"\nTotal Missed: {missed_count}")

if __name__ == "__main__":
    debug()