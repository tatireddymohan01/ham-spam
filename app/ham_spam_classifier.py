"""
ham_spam_classifier.py

Production-ready ham/spam text classifier.

Features:
- Clean, modular design (SpamClassifier class)
- Uses scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)
- Configurable paths and hyperparameters
- Proper logging and error handling
- Train / Evaluate / Predict from CLI

Usage:
    # Train model
    python ham_spam_classifier.py train \
        --data-path data/spam.csv \
        --text-column message \
        --label-column label \
        --model-path models/spam_classifier.joblib

    # Evaluate model
    python ham_spam_classifier.py evaluate \
        --data-path data/spam.csv \
        --text-column message \
        --label-column label \
        --model-path models/spam_classifier.joblib

    # Predict single message
    python ham_spam_classifier.py predict \
        --model-path models/spam_classifier.joblib \
        --text "Congratulations! You have won a free ticket!"

Requires:
    pip install scikit-learn pandas joblib
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("HamSpamClassifier")


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    c: float = 1.0
    solver: str = "liblinear"


# -----------------------------------------------------------------------------
# Core Classifier
# -----------------------------------------------------------------------------
class SpamClassifier:
    """
    Ham/Spam classifier using TF-IDF + Logistic Regression inside a sklearn Pipeline.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.pipeline: Optional[Pipeline] = None

    def build_pipeline(self) -> Pipeline:
        """Create the sklearn Pipeline."""
        logger.debug("Building model pipeline...")
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words="english",
        )
        classifier = LogisticRegression(
            C=self.config.c,
            solver=self.config.solver,
            random_state=self.config.random_state,
        )
        pipeline = Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("clf", classifier),
            ]
        )
        return pipeline

    def fit(self, X: List[str], y: List[str]) -> None:
        """Train the spam classifier."""
        logger.info("Starting training...")
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        logger.info("Training completed.")

    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for given texts."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        logger.debug("Running predictions...")
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for given texts (if supported)."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            raise RuntimeError("Underlying classifier does not support predict_proba.")
        logger.debug("Running probability predictions...")
        return self.pipeline.predict_proba(texts)

    def save(self, path: str) -> None:
        """Persist model to disk."""
        if self.pipeline is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "config": self.config}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SpamClassifier":
        """Load a saved model from disk."""
        logger.info(f"Loading model from {path}")
        data = joblib.load(path)
        obj = cls(config=data["config"])
        obj.pipeline = data["pipeline"]
        logger.info("Model successfully loaded.")
        return obj


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load_dataset(
    data_path: str, text_column: str, label_column: str
) -> Tuple[pd.Series, pd.Series]:
    """Load dataset from CSV and return text + label columns."""
    logger.info(f"Loading dataset from {data_path}")
    #df = pd.read_csv(data_path) Original line
    df = pd.read_csv(data_path,delimiter="\t",names=['label','message'])  # Modified line for tab-separated values with no header

    if text_column not in df.columns:
        raise ValueError(f"text_column '{text_column}' not found in dataset.")
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not found in dataset.")

    logger.info(f"Dataset shape: {df.shape}")
    return df[text_column].astype(str), df[label_column].astype(str)


def train_and_evaluate(
    data_path: str,
    text_column: str,
    label_column: str,
    model_path: str,
    config: TrainingConfig,
) -> None:
    """Train the model on train split, evaluate on test split, and save the model."""
    X, y = load_dataset(data_path, text_column, label_column)

    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    logger.info(
        f"Train size: {len(X_train)}, Test size: {len(X_test)}, "
        f"Positive ratio (train): {np.mean(y_train == 'spam'):.3f}"
    )

    classifier = SpamClassifier(config=config)
    classifier.fit(X_train.tolist(), y_train.tolist())

    logger.info("Evaluating on test set...")
    y_pred = classifier.predict(X_test.tolist())

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    logger.info("Confusion Matrix (rows=true, cols=pred):")
    logger.info("\n" + str(cm))

    classifier.save(model_path)


def evaluate_only(
    data_path: str,
    text_column: str,
    label_column: str,
    model_path: str,
) -> None:
    """Load an existing model and evaluate it on the provided dataset."""
    X, y = load_dataset(data_path, text_column, label_column)
    classifier = SpamClassifier.load(model_path)
    y_pred = classifier.predict(X.tolist())

    acc = accuracy_score(y, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y, y_pred))


def predict_single_text(model_path: str, text: str) -> None:
    """Load model and predict a single message."""
    classifier = SpamClassifier.load(model_path)
    pred = classifier.predict([text])[0]
    try:
        proba = classifier.predict_proba([text])[0]
        logger.info(f"Input text: {text}")
        logger.info(f"Predicted label: {pred}")
        logger.info(f"Class probabilities: {proba}")
    except RuntimeError:
        logger.info(f"Input text: {text}")
        logger.info(f"Predicted label: {pred}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ham/Spam Classifier - Training and Inference"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data-path", required=True, help="Path to CSV data file")
    train_parser.add_argument("--text-column", required=True, help="Name of text column")
    train_parser.add_argument(
        "--label-column", required=True, help="Name of label column"
    )
    train_parser.add_argument(
        "--model-path", required=True, help="Where to save trained model"
    )

    # Evaluate
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an existing model on a dataset"
    )
    eval_parser.add_argument("--data-path", required=True, help="Path to CSV data file")
    eval_parser.add_argument("--text-column", required=True, help="Name of text column")
    eval_parser.add_argument(
        "--label-column", required=True, help="Name of label column"
    )
    eval_parser.add_argument(
        "--model-path", required=True, help="Path to saved model file"
    )

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Predict for a single text")
    pred_parser.add_argument(
        "--model-path", required=True, help="Path to saved model file"
    )
    pred_parser.add_argument(
        "--text", required=True, help="Input text message to classify"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        config = TrainingConfig()  # could be customized via CLI too
        train_and_evaluate(
            data_path=args.data_path,
            text_column=args.text_column,
            label_column=args.label_column,
            model_path=args.model_path,
            config=config,
        )

    elif args.command == "evaluate":
        evaluate_only(
            data_path=args.data_path,
            text_column=args.text_column,
            label_column=args.label_column,
            model_path=args.model_path,
        )

    elif args.command == "predict":
        predict_single_text(model_path=args.model_path, text=args.text)


if __name__ == "__main__":
    main()
