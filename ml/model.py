"""
ANN Model for FAQ Intent Classification
PyTorch implementation with configurable architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class FAQDataset(Dataset):
    """Custom Dataset for FAQ data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, intent_to_idx: Dict[str, int]):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.intent_to_idx = intent_to_idx

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class IntentClassifier(nn.Module):
    """
    Neural Network for Intent Classification
    Configurable architecture with dropout and batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = None,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super(IntentClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

        logger.info(f"Model initialized: {self.get_architecture_summary()}")

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def get_architecture_summary(self) -> str:
        layers = [f"Input({self.input_dim})"]
        for dim in self.hidden_dims:
            layers.append(f"Linear({dim})")
            if self.use_batch_norm:
                layers.append("BatchNorm")
            layers.append(self.activation_name.title())
            layers.append(f"Dropout({self.dropout_rate})")
        layers.append(f"Output({self.num_classes})")
        return " -> ".join(layers)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelTrainer:
    """Handles model training, validation, and saving"""

    def __init__(
        self,
        model: IntentClassifier,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(train_loader), 100 * correct / total

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(val_loader), 100 * correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_path: str = 'saved_models/faq_intent_model.pt'
    ) -> Dict:
        logger.info(f"Starting training for {epochs} epochs...")

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if save_best:
                    self.save_model(save_path)
                    logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info("Training complete!")
        return self.history

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dims': self.model.hidden_dims,
                'num_classes': self.model.num_classes,
                'dropout_rate': self.model.dropout_rate,
                'activation': self.model.activation_name,
                'use_batch_norm': self.model.use_batch_norm
            },
            'trainer_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Model loaded from {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            axes[0].plot(self.history['train_loss'], label='Train Loss')
            axes[0].plot(self.history['val_loss'], label='Val Loss')
            axes[0].set_title('Loss')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(self.history['train_acc'], label='Train Acc')
            axes[1].plot(self.history['val_acc'], label='Val Acc')
            axes[1].set_title('Accuracy (%)')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot.")


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(class_counts)
    weights = [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
    return torch.FloatTensor(weights)
