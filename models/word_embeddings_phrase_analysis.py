import json
import spacy
import gensim 
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

nlp = spacy.load('de_core_news_sm')

def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

def create_word_embeddings(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def get_phrase_embedding(phrase, model):
    return np.mean([model.wv[word] for word in phrase if word in model.wv], axis=0)

def cluster_phrases(phrase_embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(phrase_embeddings)

def process_comments_with_embeddings(json_file, n_gram_sizes=[3, 4], n_clusters=10):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    preprocessed_comments = []
    for article in data:
        for comment_id, comment_data in article['comments'].items():
            text = comment_data.get('content')
            if text:
                preprocessed_comments.append(preprocess_text(text))

    # Create word embeddings
    model = create_word_embeddings(preprocessed_comments)

    results = {}
    for n in n_gram_sizes:
        n_grams = [tuple(comment[i:i+n]) for comment in preprocessed_comments for i in range(len(comment)-n+1)]
        n_gram_embeddings = [get_phrase_embedding(gram, model) for gram in n_grams]
        
        # Filter out zero vectors (phrases not in vocabulary)
        valid_grams = [(gram, emb) for gram, emb in zip(n_grams, n_gram_embeddings) if not np.all(emb == 0)]
        if not valid_grams:
            continue
        
        valid_n_grams, valid_embeddings = zip(*valid_grams)
        
        # Cluster phrase embeddings
        phrase_clusters = cluster_phrases(valid_embeddings, n_clusters)
        
        # Get most common phrases for each cluster
        clustered_phrases = {}
        for i in range(n_clusters):
            cluster_grams = [gram for gram, cluster in zip(valid_n_grams, phrase_clusters) if cluster == i]
            clustered_phrases[i] = Counter(cluster_grams).most_common(5)
        
        results[f'{n}-grams'] = clustered_phrases

    return results

# Usage
json_file = 'data/welt_nordstream_fixed.json'
results = process_comments_with_embeddings(json_file)

for gram_type, clusters in results.items():
    print(f"Clusters for {gram_type}:")
    for cluster_id, phrases in clusters.items():
        print(f"  Cluster {cluster_id}:")
        for phrase, count in phrases:
            print(f"    {' '.join(phrase)}: {count}")
