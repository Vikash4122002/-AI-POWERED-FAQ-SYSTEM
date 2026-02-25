"""
Test script for FAQ Vectorizer
Tests vectorization on actual FAQ data
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import TextPreprocessor
from ml.vectorizer import FAQVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    print("\n" + "=" * 70)
    print("LOADING AND PREPROCESSING FAQ DATA")
    print("=" * 70)

    df = pd.read_csv('data/faq.csv')
    print(f"Loaded {len(df)} FAQs | Unique intents: {df['intent'].nunique()}")

    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    processed_questions = preprocessor.preprocess_batch(df['question'].tolist())
    df['processed_question'] = processed_questions

    print("\nSample preprocessing:")
    for i in range(3):
        print(f"\n  Original:  {df['question'].iloc[i]}")
        print(f"  Processed: {df['processed_question'].iloc[i]}")

    return df, preprocessor


def test_vectorizer_configurations(df):
    print("\n" + "=" * 70)
    print("TESTING VECTORIZER CONFIGURATIONS")
    print("=" * 70)

    texts = df['processed_question'].tolist()

    configurations = [
        {'name': 'Basic TF-IDF (unigrams)',
         'params': {'max_features': 1000, 'ngram_range': (1, 1), 'use_idf': True}},
        {'name': 'With Bigrams',
         'params': {'max_features': 2000, 'ngram_range': (1, 2), 'use_idf': True}},
        {'name': 'No IDF (TF only)',
         'params': {'max_features': 1000, 'ngram_range': (1, 1), 'use_idf': False}},
        {'name': 'With dimensionality reduction (SVD)',
         'params': {'max_features': 5000, 'ngram_range': (1, 2), 'use_idf': True,
                    'reduce_dimensions': True, 'n_components': 20}},
    ]

    results = []
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        vectorizer = FAQVectorizer(**config['params'])
        X = vectorizer.fit_transform(texts)
        stats = vectorizer.analyze_vocabulary(texts)

        print(f"  Shape: {X.shape} | Vocab: {stats['vocabulary_size']} | "
              f"Sparsity: {stats['sparsity']:.4f}")

        if vectorizer.use_idf and not config['params'].get('reduce_dimensions'):
            top = vectorizer.get_top_features(3)
            print(f"  Top 3 distinctive terms: {[(t, round(s, 3)) for t, s in top]}")

        results.append({'name': config['name'], 'shape': X.shape, 'vectorizer': vectorizer})

    return results


def test_similarity_matching(df, vectorizer):
    print("\n" + "=" * 70)
    print("TESTING SIMILARITY MATCHING")
    print("=" * 70)

    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    texts = df['processed_question'].tolist()
    questions = df['question'].tolist()
    intents = df['intent'].tolist()

    X = vectorizer.transform(texts)
    if hasattr(X, 'toarray'):
        X = X.toarray()

    test_queries = [
        "How do I reset my password?",
        "I want to delete my account",
        "Can I get my money back?",
        "How to contact customer service",
        "What's the price of premium?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        processed = preprocessor.preprocess(query)
        query_vec = vectorizer.transform([processed])
        if hasattr(query_vec, 'toarray'):
            query_vec = query_vec.toarray()

        sims = cosine_similarity(query_vec, X)[0]
        top3 = np.argsort(sims)[-3:][::-1]

        print("  Top matches:")
        for idx in top3:
            if sims[idx] > 0.05:
                print(f"    [{intents[idx]}] {questions[idx][:55]}... (sim: {sims[idx]:.3f})")


def test_save_load(vectorizer):
    print("\n" + "=" * 70)
    print("TESTING SAVE/LOAD")
    print("=" * 70)

    os.makedirs('saved_models', exist_ok=True)
    save_path = 'saved_models/test_vectorizer_tmp.pkl'
    vectorizer.save(save_path)

    new_vec = FAQVectorizer()
    new_vec.load(save_path)
    print(f"  Original vocab: {vectorizer.vocabulary_size} | Loaded vocab: {new_vec.vocabulary_size}")
    assert vectorizer.vocabulary_size == new_vec.vocabulary_size, "Vocab mismatch!"
    print("  Save/load verified successfully!")
    os.remove(save_path)


def main():
    df, preprocessor = load_and_preprocess_data()
    results = test_vectorizer_configurations(df)
    best_vectorizer = results[1]['vectorizer']  # bigrams config
    test_similarity_matching(df, best_vectorizer)
    test_save_load(best_vectorizer)
    print("\n" + "=" * 70)
    print("ALL VECTORIZER TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
