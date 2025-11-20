import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    narrative_sources = []
    
    # --- SOURCE 1: High-Quality Curated Narratives ---
    # These are the "perfect" sentences you uploaded
    files = ['disinfo_titles_cleaned.csv', 'more_disinfo.csv']
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            if 'disinfo_title' in df.columns:
                narrative_sources.append(df['disinfo_title'])
                print(f"Loaded {len(df)} narratives from {f}")

    # --- SOURCE 2: The "Subset" from your Dataset ---
    # We pull the actual text from the disinformation examples in your dataset
    # so the bot learns to recognize them.
    examples_file = 'data/all_223_examples.csv'
    if os.path.exists(examples_file):
        df_examples = pd.read_csv(examples_file)
        
        # Filter for DISINFO only (Label = 1)
        disinfo_subset = df_examples[df_examples['label'] == 1].copy()
        
        # Use translated text if available, else raw text
        if 'translated_text' in disinfo_subset.columns:
            disinfo_subset['narrative_text'] = disinfo_subset['translated_text'].fillna(disinfo_subset['text'])
        else:
            disinfo_subset['narrative_text'] = disinfo_subset['text']
            
        # Clean up: remove short texts that might be noise
        # (We only want substantial claims, e.g., > 20 chars)
        disinfo_subset = disinfo_subset[disinfo_subset['narrative_text'].str.len() > 20]
        
        narrative_sources.append(disinfo_subset['narrative_text'])
        print(f"Injecting {len(disinfo_subset)} examples from {examples_file} into the Knowledge Base.")
    else:
        print(f"Warning: {examples_file} not found. Skipping subset injection.")

    # --- 3. Merge and Save Knowledge Base ---
    if narrative_sources:
        # Concatenate all sources
        combined_series = pd.concat(narrative_sources, ignore_index=True)
        
        # Deduplicate and Format
        kb_df = pd.DataFrame({'narrative': combined_series})
        kb_df = kb_df.drop_duplicates().dropna()
        
        # Save to the file the Labeler reads
        kb_df.to_csv('data/known_narratives.csv', index=False)
        print(f"SUCCESS: Created 'data/known_narratives.csv' with {len(kb_df)} total narratives.")
    else:
        print("Error: No narrative sources found!")

    # --- 4. Create Tuning/Test Splits (Standard) ---
    if os.path.exists(examples_file):
        df = pd.read_csv(examples_file)
        if 'translated_text' in df.columns:
            df['translated_text'] = df['translated_text'].fillna(df['text'])
        
        tuning_set, test_set = train_test_split(df, test_size=0.7, random_state=42, stratify=df['label'])
        
        tuning_set.to_csv('data/tuning_data.csv', index=False)
        test_set.to_csv('data/test_data.csv', index=False)
        print(f"Data Splits: tuning_data.csv ({len(tuning_set)}), test_data.csv ({len(test_set)})")

if __name__ == "__main__":
    prepare_data()