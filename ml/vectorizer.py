"""
Text Vectorization Module for FAQ System
Converts preprocessed text to numerical features using TF-IDF
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAQVectorizer:
    """
    Vectorizer class for converting text to TF-IDF features
    Handles training, saving, loading, and transformation
    """

    def __init__(
        self,
        max_features: int = 5000,
        max_df: float = 0.8,
        min_df: int = 1,
        ngram_range: Tuple[int, int] = (1, 2),
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = True,
        reduce_dimensions: bool = False,
        n_components: int = 100
    ):
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.reduce_dimensions = reduce_dimensions
        self.n_components = n_components

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            token_pattern=r'(?u)\b\w+\b',
            strip_accents='unicode'
        )

        self.svd = None
        if reduce_dimensions:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)

        self.is_fitted = False
        self.vocabulary_size = 0
        self.feature_names = None

        logger.info(f"FAQVectorizer initialized with max_features={max_features}, "
                    f"ngram_range={ngram_range}")

    def fit(self, texts: List[str]) -> 'FAQVectorizer':
        logger.info(f"Fitting vectorizer on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"Vectorizer fitted. Vocabulary size: {self.vocabulary_size}")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        X = self.vectorizer.transform(texts)

        if self.reduce_dimensions and self.svd is not None:
            if not hasattr(self.svd, 'components_'):
                logger.info(f"Fitting SVD with {self.n_components} components...")
                self.svd.fit(X)
            X = self.svd.transform(X)

        return X

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.feature_names

    def get_vocabulary(self) -> Dict:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.vocabulary_

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'is_fitted': self.is_fitted,
            'vocabulary_size': self.vocabulary_size,
            'max_features': self.max_features,
            'max_df': self.max_df,
            'min_df': self.min_df,
            'ngram_range': self.ngram_range,
            'use_idf': self.use_idf,
            'smooth_idf': self.smooth_idf,
            'sublinear_tf': self.sublinear_tf,
            'reduce_dimensions': self.reduce_dimensions,
            'n_components': self.n_components
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Vectorizer saved to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")

        save_dict = joblib.load(filepath)
        self.vectorizer = save_dict['vectorizer']
        self.svd = save_dict['svd']
        self.is_fitted = save_dict['is_fitted']
        self.vocabulary_size = save_dict['vocabulary_size']
        self.max_features = save_dict['max_features']
        self.max_df = save_dict['max_df']
        self.min_df = save_dict['min_df']
        self.ngram_range = save_dict['ngram_range']
        self.use_idf = save_dict['use_idf']
        self.smooth_idf = save_dict['smooth_idf']
        self.sublinear_tf = save_dict['sublinear_tf']
        self.reduce_dimensions = save_dict['reduce_dimensions']
        self.n_components = save_dict['n_components']

        if self.is_fitted:
            self.feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"Vectorizer loaded from {filepath}")
        logger.info(f"Vocabulary size: {self.vocabulary_size}")

    def analyze_vocabulary(self, texts: List[str]) -> Dict:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")

        X = self.transform(texts)

        stats = {
            'vocabulary_size': self.vocabulary_size,
            'sparsity': 1 - (np.count_nonzero(X.toarray() if hasattr(X, 'toarray') else X) /
                             (X.shape[0] * X.shape[1])),
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }

        if self.reduce_dimensions and self.svd is not None:
            stats['svd_components'] = self.n_components
            if hasattr(self.svd, 'explained_variance_ratio_'):
                stats['svd_explained_variance'] = self.svd.explained_variance_ratio_.sum()

        return stats

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if not self.is_fitted or not hasattr(self.vectorizer, 'idf_'):
            raise ValueError("Vectorizer must be fitted with use_idf=True")

        idf_scores = self.vectorizer.idf_
        feature_names = self.get_feature_names()
        top_indices = np.argsort(idf_scores)[-n:][::-1]
        return [(feature_names[i], idf_scores[i]) for i in top_indices]


if __name__ == "__main__":
    from ml.preprocess import TextPreprocessor

    sample_texts = [
        "reset password forgot",
        "delete account close",
        "refund policy return money",
        "contact support help",
    ]

    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    processed = preprocessor.preprocess_batch(sample_texts)

    vectorizer = FAQVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed)
    print(f"Feature matrix shape: {X.shape}")
