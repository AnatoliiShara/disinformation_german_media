import json 
import spacy
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# load German model
nlp = spacy.load("de_core_news_sm")

# load German tokenizer and model
model_name = "dbmdz/bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)   

sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')   

def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]   


def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Reshape embeddings to 2D format
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1)

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(embeddings)

def process_comments(json_file, ngram_sizes=[3,4], top_n=10, n_clusters=5):
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    all_comments = [comment['content'] for article in data for comment in article['comments'].values()]
    
    # Change results to a dictionary
    results = {}
    
    for n in ngram_sizes:
        all_ngrams = []
        for comment in all_comments:
            tokens = preprocess_text(comment)
            all_ngrams.extend(extract_ngrams(tokens, n))
            
        # get most frequent ngrams
        ngram_freq = Counter(all_ngrams)
        top_ngrams = ngram_freq.most_common(top_n)
        results[f"top_{n}-grams"] = top_ngrams
        
        # Cluster ngrams using BERT embeddings
        ngram_embeddings = np.vstack([get_bert_embeddings([ngram]) for ngram, _ in top_ngrams])
        clusters = cluster_embeddings(ngram_embeddings, n_clusters)
        
        cluster_ngrams = {}
        for i, (ngram, count) in enumerate(top_ngrams):
            cluster = clusters[i]
            if cluster not in cluster_ngrams:
                cluster_ngrams[cluster] = []
            cluster_ngrams[cluster].append((ngram, count))
        results[f"clustered_{n}-grams"] = cluster_ngrams
        
    # Analyze comment similarity using SentenceTransformer
    comment_embeddings = sentence_model.encode(all_comments)
    comment_clusters = cluster_embeddings(comment_embeddings, n_clusters)
    
    clustered_comments = {}
    for i, comment in enumerate(all_comments):
        cluster = comment_clusters[i]
        if cluster not in clustered_comments:
            clustered_comments[cluster] = []
        clustered_comments[cluster].append(comment[:100] + ". . .")
    
    results['comment_clusters'] = clustered_comments
    return results

json_file = "data/welt_nordstream_fixed.json"    
results = process_comments(json_file)

print("TOP N-GRAMS")
for gram_type, ngrams in results.items():
    if gram_type.startswith("top_"):
        print(f"\n{gram_type}:")
        for gram, count in ngrams:
            print(f"   {gram}: {count}")
            
print("\nClustered N-GRAMS")
for gram_type, clusters in results.items():
    if gram_type.startswith("clustered_"):
        print(f"\n{gram_type}")
        for cluster, ngrams in clusters.items():
            print(f"   Cluster: {cluster}:")
            for gram, count in ngrams:
                print(f"    {gram} : {count}")
            
print("\nComment Clusters")
for cluster, comments in results['comment_clusters'].items():
    print(f"\nCluster {cluster}")
    for comment in comments[:5]:
        print(f"   {comment}")
