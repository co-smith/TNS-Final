import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    print("Loading data...")
    df = pd.read_csv('data/all_223_examples.csv')
    
    df['final_text'] = df['translated_text'].fillna(df['text'])

    # We take 50% of the disinformation to build our "Knowledge Base"
    # The rest go into the "Test Set"
    
    disinfo_df = df[df['label'] == 1]
    safe_df = df[df['label'] == 0]
    
    narratives_train, disinfo_test = train_test_split(disinfo_df, test_size=0.5, random_state=42)
    
    # Combine the test sets
    test_set = pd.concat([disinfo_test, safe_df]).sample(frac=1, random_state=42) 
    
    # Save the Knowledge Base
    print(f"Extracting {len(narratives_train)} narratives for the Knowledge Base...")
    narratives_train[['final_text']].to_csv('data/known_narratives.csv', index=False, header=['narrative'])
    
    # Save the Test Set
    test_set.to_csv('data/test_data.csv', index=False)
    
    print("Files 'known_narratives.csv' and 'test_data.csv' created.")

if __name__ == "__main__":
    prepare_data()