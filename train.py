"""
Training Script for FAQ Intent Classifier
Run this script to train and save the model and vectorizer
Usage: python train.py
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import os
import json
from collections import Counter

from ml.preprocess import TextPreprocessor
from ml.vectorizer import FAQVectorizer
from ml.model import IntentClassifier, ModelTrainer, FAQDataset, compute_class_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data():
    """Load, preprocess and vectorize FAQ data"""

    print("\n" + "=" * 70)
    print("PREPARING FAQ DATA FOR TRAINING")
    print("=" * 70)

    df = pd.read_csv('data/faq.csv')
    print(f"Loaded {len(df)} FAQs | Unique intents: {df['intent'].nunique()}")

    # Preprocess
    preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    processed_questions = preprocessor.preprocess_batch(df['question'].tolist())

    # Intent mappings
    unique_intents = sorted(df['intent'].unique())
    intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}
    labels = [intent_to_idx[intent] for intent in df['intent']]

    print("\nIntent mapping:")
    for intent, idx in intent_to_idx.items():
        count = labels.count(idx)
        print(f"  {idx}: {intent} ({count} samples)")

    # Vectorize
    vectorizer = FAQVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        use_idf=True,
        min_df=1,
        reduce_dimensions=False
    )
    features = vectorizer.fit_transform(processed_questions)
    print(f"\nFeature matrix shape: {features.shape}")

    return features, labels, intent_to_idx, idx_to_intent, vectorizer, preprocessor


def train():
    """Main training function"""

    print("\n" + "=" * 70)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 70)

    os.makedirs('saved_models', exist_ok=True)

    # Prepare data
    features, labels, intent_to_idx, idx_to_intent, vectorizer, preprocessor = prepare_data()

    # Convert to arrays
    X = features.toarray() if hasattr(features, 'toarray') else features
    y = np.array(labels)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

    # Datasets and loaders
    train_dataset = FAQDataset(X_train, y_train, intent_to_idx)
    val_dataset = FAQDataset(X_val, y_val, intent_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Class weights
    class_weights = compute_class_weights(y_train.tolist())

    # Model
    model = IntentClassifier(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64],
        num_classes=len(intent_to_idx),
        dropout_rate=0.3,
        activation='relu',
        use_batch_norm=True
    )
    print(f"\nArchitecture: {model.get_architecture_summary()}")
    print(f"Total parameters: {model.count_parameters():,}")

    # Trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5,
        class_weights=class_weights
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_best=True,
        save_path='saved_models/faq_intent_model.pt'
    )

    # Save vectorizer
    vectorizer.save('saved_models/vectorizer.pkl')

    # Save intent mappings
    mappings = {'intent_to_idx': intent_to_idx, 'idx_to_intent': idx_to_intent}
    with open('saved_models/intent_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)

    print(f"\nBest val loss:     {min(history['val_loss']):.4f}")
    print(f"Best val accuracy: {max(history['val_acc']):.2f}%")
    print("\nAll artifacts saved to saved_models/")

    # Optional: plot
    trainer.plot_training_history(save_path='saved_models/training_history.png')


if __name__ == "__main__":
    train()
