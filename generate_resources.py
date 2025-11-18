import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    kb_df = pd.read_csv('data/Disinformation_Russia_Ukraine_3months.csv', encoding='cp1252')
    kb_df[['Disinfo']].rename(columns={'Disinfo': 'narrative'}).dropna().to_csv('data/known_narratives.csv', index=False)

    # Build Validation and Test Sets
    df = pd.read_csv('data/all_223_examples.csv')
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    # 30% Validation (Tuning), 70% Test
    tuning_set, test_set = train_test_split(df, test_size=0.7, random_state=42, stratify=df['label'])
    
    tuning_set.to_csv('data/tuning_data.csv', index=False)
    test_set.to_csv('data/test_data.csv', index=False)
    
    print(f"Files created: known_narratives.csv, tuning_data.csv ({len(tuning_set)}), test_data.csv ({len(test_set)})")

if __name__ == "__main__":
    prepare_data()