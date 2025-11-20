import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    # Make sure data folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    narrative_sources = []
    
    #SPLIT DATA

    # We do this first to prevent data leakage
    df = pd.read_csv('data/all_223_examples.csv')
    
    # Fill empty translation columns
    df['translated_text'] = df['translated_text'].fillna(df['text'])
    
    # Split 70% for tuning, 30% for final testing
    # Stratify makes sure we have equal ratio of disinfo in both
    tuning_set, test_set = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    
    # Save these files so other scripts use the exact same split
    tuning_set.to_csv('data/tuning_data.csv', index=False)
    test_set.to_csv('data/test_data.csv', index=False)

    #BUILD KNOWLEDGE BASE
    
    # Source A: The curated files
    files = ['disinfo_titles_cleaned.csv', 'more_disinfo.csv']
    for f in files:
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            narrative_sources.append(temp_df['disinfo_title'])
            print(f"Added external narratives from {f}")

    # Source B: The Disinfo examples from our Tuning Set
    # Filter for label=1 (Disinfo)
    disinfo_subset = tuning_set[tuning_set['label'] == 1].copy()
    
    # Use the translated text
    disinfo_subset['narrative_text'] = disinfo_subset['translated_text'].fillna(disinfo_subset['text'])
    
    # Ignore really short strings (noise)
    clean_subset = disinfo_subset[disinfo_subset['narrative_text'].str.len() > 20]
    
    narrative_sources.append(clean_subset['narrative_text'])
    print(f"Added {len(clean_subset)} examples from Tuning Set to KB")

    # MERGE AND SAVE
    # Combine everything into one big list
    combined_series = pd.concat(narrative_sources, ignore_index=True)
    
    # Make dataframe, drop duplicates
    kb_df = pd.DataFrame({'narrative': combined_series})
    kb_df = kb_df.drop_duplicates().dropna()
    
    # Save to CSV for the labeler to read
    kb_df.to_csv('data/known_narratives.csv', index=False)
    print(f"Knowledge Base saved! Total narratives: {len(kb_df)}")

if __name__ == "__main__":
    prepare_data()