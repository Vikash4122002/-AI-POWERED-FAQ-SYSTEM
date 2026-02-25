"""
Test script to verify preprocessing on the FAQ dataset
"""

import pandas as pd
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_preprocessing_on_faqs():
    """Test the preprocessor on actual FAQ data"""

    print("\n" + "=" * 70)
    print("TESTING PREPROCESSOR ON FAQ DATASET")
    print("=" * 70)

    df = pd.read_csv('data/faq.csv')
    logger.info(f"Loaded {len(df)} FAQs from dataset")

    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

    unique_intents = df['intent'].unique()
    print("\nPREPROCESSING SAMPLES BY INTENT:")
    print("-" * 70)

    for intent in unique_intents[:5]:
        sample = df[df['intent'] == intent].iloc[0]
        original = sample['question']
        processed = preprocessor.preprocess(original)
        summary = preprocessor.get_preprocessing_summary(original, processed)

        print(f"\nIntent: {intent}")
        print(f"  Original:  {original}")
        print(f"  Processed: {processed}")
        print(f"  Stats: {summary['original_word_count']} words -> "
              f"{summary['processed_word_count']} words "
              f"({summary['reduction_percentage']:.1f}% reduction)")

    print("\n" + "=" * 70)
    print("TESTING BATCH PROCESSING")
    print("=" * 70)

    sample_questions = df['question'].sample(10, random_state=42).tolist()
    processed_batch = preprocessor.preprocess_batch(sample_questions)

    for i, (orig, proc) in enumerate(zip(sample_questions, processed_batch), 1):
        print(f"\n{i}. Original:  {orig[:60]}")
        print(f"   Processed: {proc[:60]}")

    print("\n" + "=" * 70)
    print("PERFORMANCE TEST")
    print("=" * 70)

    all_questions = df['question'].tolist()
    start = time.time()
    processed_all = preprocessor.preprocess_batch(all_questions)
    elapsed = time.time() - start

    print(f"\nProcessed {len(all_questions)} questions in {elapsed:.2f}s")
    print(f"Average: {elapsed / len(all_questions) * 1000:.2f} ms/question")

    all_words = ' '.join(processed_all).split()
    unique_words = set(all_words)
    print(f"\nVocabulary Stats:")
    print(f"  Total words:  {len(all_words)}")
    print(f"  Unique words: {len(unique_words)}")
    print(f"  Compression:  {len(all_words)/len(unique_words):.2f}:1")


def compare_preprocessing_methods():
    """Compare different preprocessing approaches"""

    print("\n" + "=" * 70)
    print("COMPARING PREPROCESSING METHODS")
    print("=" * 70)

    question = "How can I reset my password? I've forgotten it!!! Please help ASAP."

    configs = [
        {"name": "Minimal (lowercase only)", "use_lemmatization": False, "remove_stopwords": False},
        {"name": "Stopword removal only", "use_lemmatization": False, "remove_stopwords": True},
        {"name": "Lemmatization only", "use_lemmatization": True, "remove_stopwords": False},
        {"name": "Full pipeline (lemmatization + stopwords)", "use_lemmatization": True, "remove_stopwords": True},
    ]

    print(f"\nOriginal: {question}\n")

    for config in configs:
        preprocessor = TextPreprocessor(
            use_lemmatization=config["use_lemmatization"],
            remove_stopwords=config["remove_stopwords"]
        )
        processed = preprocessor.preprocess(question)
        print(f"{config['name']}:")
        print(f"  -> {processed}\n")


if __name__ == "__main__":
    test_preprocessing_on_faqs()
    compare_preprocessing_methods()
