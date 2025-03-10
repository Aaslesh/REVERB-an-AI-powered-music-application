import pandas as pd
import h5py
import numpy as np
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Install python-Levenshtein for faster matching
# pip install python-Levenshtein

def load_data():
    try:
        # Try loading preprocessed data
        with h5py.File('recommend.h5', 'r') as hf:
            # Load DataFrame columns
            df = pd.DataFrame({
                'artist': [x.decode() for x in hf['artist'][:]],
                'song': [x.decode() for x in hf['song'][:]],
                'text': [x.decode() for x in hf['text'][:]]
            })
            
            # Load sparse matrix components
            matrix = csr_matrix((
                hf['matrix_data'][:],
                hf['matrix_indices'][:],
                hf['matrix_indptr'][:]
            ), shape=hf['matrix_shape'][:])
            
            print("Loaded preprocessed data from HDF5")
            return df, matrix
        
    except (FileNotFoundError, KeyError):
        print("First-time setup: Processing dataset...")
        # Load only necessary columns with optimized memory
        df = pd.read_csv("spotify_millsongdata.csv", 
                       usecols=['artist', 'song', 'text'],
                       dtype={'artist': 'category', 
                             'song': 'category',
                             'text': 'string'})
        
        # Clean and preprocess text
        df['text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df['song'] = df['song'].str.lower().str.strip().astype('string')
        
        # Create TF-IDF matrix with limited features
        tfidf = TfidfVectorizer(stop_words='english', 
                               max_features=1500,
                               dtype=np.float32)
        matrix = tfidf.fit_transform(df['text'])
        
        # Save to HDF5 with compression
        with h5py.File('recommend.h5', 'w') as hf:
            # Store string data as fixed-length bytes
            hf.create_dataset('artist', 
                            data=df['artist'].astype('S'), 
                            compression='gzip')
            hf.create_dataset('song', 
                            data=df['song'].astype('S'), 
                            compression='gzip')
            hf.create_dataset('text', 
                            data=df['text'].astype('S'), 
                            compression='gzip')
            
            # Store sparse matrix components
            hf.create_dataset('matrix_data', 
                             data=matrix.data.astype(np.float32),
                             compression='gzip')
            hf.create_dataset('matrix_indices', 
                             data=matrix.indices,
                             compression='gzip')
            hf.create_dataset('matrix_indptr', 
                             data=matrix.indptr,
                             compression='gzip')
            hf.create_dataset('matrix_shape', 
                             data=matrix.shape,
                             compression='gzip')
        
        print("Created new HDF5 dataset")
        return df, matrix

def find_song(song_name, df):
    song_lower = song_name.lower().strip()
    matches = process.extract(song_lower, 
                            df['song'].str.lower(), 
                            limit=3)
    return next((idx for match, score, idx in matches if score > 70), None)

def get_recommendations(song_idx, matrix, df, top_n=5):
    # Calculate cosine similarities efficiently
    query_vector = matrix[song_idx]
    # Convert sparse matrix to array before flattening
    scores = matrix.dot(query_vector.T).toarray().flatten()
    top_indices = scores.argsort()[::-1][1:top_n+1]
    return df.iloc[top_indices]

if __name__ == "__main__":
    df, tfidf_matrix = load_data()
    
    # Format display text
    df['song'] = df['song'].str.title()
    df['artist'] = df['artist'].str.title()
    
    print("\n=== Music Recommendation System ===")
    print("Enter 'q' to quit\n")
    
    while True:
        song_query = input("Enter song name: ").strip()
        if song_query.lower() == 'q':
            print("\nGoodbye! Happy listening!")
            break
        
        song_idx = find_song(song_query, df)
        
        if song_idx is not None:
            recommendations = get_recommendations(song_idx, tfidf_matrix, df)
            print(f"\nRecommendations for '{song_query.title()}':")
            print("-" * 40)
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {row['song']}")
                print(f"   Artist: {row['artist']}")
                print("-" * 40)
        else:
            print(f"\nSong not found. Did you mean:")
            matches = process.extract(song_query, df['song'], limit=3)
            for match, score, _ in matches:
                if score > 50:
                    print(f"- {match} ({score}% match)")