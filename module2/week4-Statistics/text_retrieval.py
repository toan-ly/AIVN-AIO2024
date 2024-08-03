import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    vi_data_df = pd.read_csv(file_path)
    context = vi_data_df['text']
    context = [doc.lower() for doc in context]
    return context, vi_data_df

def tfidf_vectorize(context):
    tfidf_vectorizer = TfidfVectorizer()
    context_embedded = tfidf_vectorizer.fit_transform(context)
    return tfidf_vectorizer, context_embedded

def tfidf_search(question, tfidf_vectorizer, context_embedded, top_d=5):
    # lowercasing before encoding
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    cosine_scores = cosine_similarity(query_embedded, context_embedded).flatten()
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_scores = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_scores)
    return results

def corr_search(question, tfidf_vectorizer, context_embedded, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    corr_scores = np.corrcoef(query_embedded.toarray(), context_embedded.toarray())[0][1:]
    
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)
    return results