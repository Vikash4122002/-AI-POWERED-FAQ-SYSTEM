"""
Inference logic for FAQ system
Handles the prediction pipeline: preprocess -> vectorize -> classify -> retrieve answer
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
import time
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import IntentClassifier
from ml.preprocess import TextPreprocessor
from ml.vectorizer import FAQVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FAQInference:
    """
    Inference class for the FAQ system.
    Loads trained model + vectorizer and exposes prediction methods.
    """

    def __init__(
        self,
        model_path: str = 'saved_models/faq_intent_model.pt',
        vectorizer_path: str = 'saved_models/vectorizer.pkl',
        faq_data_path: str = 'data/faq.csv',
        mappings_path: str = 'saved_models/intent_mappings.json'
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.faq_data_path = faq_data_path
        self.mappings_path = mappings_path

        self.preprocessor = None
        self.vectorizer = None
        self.model = None
        self.intent_to_idx = None
        self.idx_to_intent = None
        self.faq_df = None
        self.faq_vectors = None
        self.faq_questions = None
        self.faq_answers = None
        self.faq_intents = None

        self._load_components()
        logger.info("FAQInference initialized successfully")

    def _load_components(self):
        """Load all pipeline components"""
        logger.info("Loading preprocessor...")
        self.preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

        logger.info(f"Loading vectorizer from {self.vectorizer_path}...")
        self.vectorizer = FAQVectorizer()
        self.vectorizer.load(self.vectorizer_path)

        logger.info(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        model_config = checkpoint['model_config']

        self.model = IntentClassifier(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation'],
            use_batch_norm=model_config['use_batch_norm']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self._load_intent_mappings()
        self._load_faq_data()
        self._precompute_faq_vectors()

    def _load_intent_mappings(self):
        if os.path.exists(self.mappings_path):
            with open(self.mappings_path, 'r') as f:
                mappings = json.load(f)
            self.intent_to_idx = mappings['intent_to_idx']
            self.idx_to_intent = {int(k): v for k, v in mappings['idx_to_intent'].items()}
        else:
            df = pd.read_csv(self.faq_data_path)
            unique_intents = sorted(df['intent'].unique())
            self.intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
            self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        logger.info(f"Loaded {len(self.intent_to_idx)} intent classes")

    def _load_faq_data(self):
        self.faq_df = pd.read_csv(self.faq_data_path)
        self.faq_questions = self.faq_df['question'].tolist()
        self.faq_answers = self.faq_df['answer'].tolist()
        self.faq_intents = self.faq_df['intent'].tolist()
        logger.info(f"Loaded {len(self.faq_questions)} FAQ entries")

    def _precompute_faq_vectors(self):
        logger.info("Precomputing FAQ vectors...")
        processed = self.preprocessor.preprocess_batch(self.faq_questions)
        self.faq_vectors = self.vectorizer.transform(processed)
        if hasattr(self.faq_vectors, 'toarray'):
            self.faq_vectors = self.faq_vectors.toarray()
        logger.info(f"Precomputed {len(self.faq_vectors)} FAQ vectors")

    def predict_intent(self, question: str) -> Tuple[str, float, np.ndarray]:
        processed = self.preprocessor.preprocess(question)
        features = self.vectorizer.transform([processed])
        if hasattr(features, 'toarray'):
            features = features.toarray()

        X = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])

        return self.idx_to_intent[predicted_idx], confidence, probs

    def find_best_answer(self, question: str, intent: Optional[str] = None) -> Tuple[str, str, float]:
        processed = self.preprocessor.preprocess(question)
        query_vec = self.vectorizer.transform([processed])
        if hasattr(query_vec, 'toarray'):
            query_vec = query_vec.toarray()

        if intent and intent in self.intent_to_idx:
            intent_indices = [i for i, it in enumerate(self.faq_intents) if it == intent]
        else:
            intent_indices = list(range(len(self.faq_intents)))

        faq_vecs = self.faq_vectors[intent_indices]
        sims = cosine_similarity(query_vec, faq_vecs)[0]
        best_local_idx = int(np.argmax(sims))
        best_idx = intent_indices[best_local_idx]

        return (
            self.faq_answers[best_idx],
            self.faq_intents[best_idx],
            float(sims[best_local_idx])
        )

    def get_answer(self, question: str) -> Dict:
        start = time.time()

        predicted_intent, confidence, all_probs = self.predict_intent(question)
        answer, matched_intent, similarity = self.find_best_answer(question, intent=predicted_intent)

        return {
            'question': question,
            'answer': answer,
            'intent': matched_intent,
            'confidence': confidence,
            'similarity_score': similarity,
            'processing_time_ms': (time.time() - start) * 1000,
            'all_probabilities': {
                self.idx_to_intent[i]: float(p) for i, p in enumerate(all_probs)
            }
        }

    def get_batch_answers(self, questions: List[str]) -> List[Dict]:
        results = []
        for question in questions:
            try:
                results.append(self.get_answer(question))
            except Exception as e:
                logger.error(f"Error processing '{question}': {e}")
                results.append({
                    'question': question,
                    'answer': "Error processing question. Please try again.",
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'error': str(e)
                })
        return results

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'ANN (PyTorch)',
            'input_dim': self.model.input_dim,
            'hidden_layers': self.model.hidden_dims,
            'num_classes': self.model.num_classes,
            'classes': list(self.idx_to_intent.values()),
            'vectorizer_params': {
                'max_features': self.vectorizer.max_features,
                'ngram_range': list(self.vectorizer.ngram_range),
                'use_idf': self.vectorizer.use_idf
            },
            'total_parameters': self.model.count_parameters(),
            'device': str(device)
        }
