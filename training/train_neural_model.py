"""
Training script for neural sentiment models with MLflow tracking.

This script extracts the core training pipeline originally built in the
`P7_neural_network.ipynb` notebook so it can be executed headlessly (CI/CD,
scheduled jobs, etc.). By default it reproduces the production Bidirectional GRU
with GloVe embeddings and logs the run to the configured MLflow tracking server.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib

# Use a non-interactive backend so plots can be generated in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.keras  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_hub as hub  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gensim.models import FastText, Word2Vec  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tensorflow.keras import regularizers  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    GRU,
    InputLayer,
    LSTM,
    RepeatVector,
    SimpleRNN,
)  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")


class NeuralTweetClassifier:
    """
    Neural network classifier supporting multiple embeddings and architectures.

    The implementation mirrors the notebook version but is packaged for reuse in
    scripts and CI pipelines.
    """

    def __init__(
        self,
        embedding: str,
        embedding_dim: int = 100,
        units: int = 128,
        max_length: int = 100,
        model_type: str = "LSTM",
        base_dir: str | Path = ".",
        register_model_name: Optional[str] = None,
    ) -> None:
        self.embedding = embedding
        self.embedding_dim = 512 if embedding == "use" else embedding_dim
        self.units = units
        self.max_length = max_length
        self.model_type = model_type
        self.model: Optional[tf.keras.Model] = None
        self.use_encoder = None
        self.register_model_name = register_model_name

        if embedding == "use":
            self.tokenizer: Optional[Tokenizer] = None
        else:
            self.tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

        self.base_dir = Path(base_dir)
        self.artifacts_dir = self.base_dir / "artifacts"
        self.models_dir = self.base_dir / "models"
        self.checkpoints_dir = self.base_dir / "checkpoints"

        valid_model_types = [
            "LSTM",
            "RNN",
            "Bidirectional_LSTM",
            "GRU",
            "Bidirectional_GRU",
            "Dense",
        ]
        valid_embeddings = ["w2v", "fasttext", "glove", "use"]

        if model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}")
        if embedding not in valid_embeddings:
            raise ValueError(f"embedding must be one of {valid_embeddings}")

    # === TOKENIZER ===
    def fit_tokenizer(self, texts: pd.Series) -> np.ndarray:
        """Tokenise the input texts and return padded sequences."""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, maxlen=self.max_length, padding="post", truncating="post"
        )
        return padded_sequences

    # === EMBEDDING ===
    def build_embedding_matrix(self, texts: pd.Series, use_pretrained: bool = True) -> np.ndarray:
        """Build an embedding matrix for the current tokenizer vocabulary."""
        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        if self.embedding == "glove":
            import gensim.downloader as api

            model = api.load("glove-twitter-100")
            has_wv_attr = hasattr(model, "wv")
        elif self.embedding == "w2v" and use_pretrained:
            try:
                model = Word2Vec.load(f"../models/pretrained_w2v_{self.embedding_dim}.model")
                has_wv_attr = hasattr(model, "wv")
            except Exception:
                model = Word2Vec(
                    sentences=[s.split() for s in texts],
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=1,
                    workers=4,
                )
                has_wv_attr = hasattr(model, "wv")
                model.save(f"../models/pretrained_w2v_{self.embedding_dim}.model")
        elif self.embedding == "fasttext":
            try:
                model = FastText.load(f"../models/pretrained_fasttext_{self.embedding_dim}.model")
                has_wv_attr = hasattr(model, "wv")
            except FileNotFoundError:
                model = FastText(
                    sentences=[s.split() for s in texts],
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=1,
                    workers=4,
                )
                has_wv_attr = hasattr(model, "wv")
                model.save(f"../models/pretrained_fasttext_{self.embedding_dim}.model")
        else:
            model = Word2Vec(
                sentences=[s.split() for s in texts],
                vector_size=self.embedding_dim,
                window=5,
                min_count=1,
                workers=4,
            )
            has_wv_attr = hasattr(model, "wv")

        for word, index in word_index.items():
            if has_wv_attr and word in model.wv:
                embedding_matrix[index] = model.wv[word]
            elif not has_wv_attr and word in model:
                embedding_matrix[index] = model[word]

        return embedding_matrix

    def _load_use_encoder(self):
        if self.use_encoder is None:
            self.use_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        return self.use_encoder

    def _encode_with_use(self, texts: pd.Series, batch_size: int = 64):
        encoder = self._load_use_encoder()
        if isinstance(texts, (list, tuple)):
            texts = pd.Series(texts)
        texts = texts.fillna("")
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts.iloc[start : start + batch_size].tolist()
            embeddings.append(encoder(batch).numpy())
        return np.vstack(embeddings)

    # === MODEL ===
    def build_model(self, vocab_size: Optional[int], embedding_matrix: Optional[np.ndarray]):
        """Build the selected architecture."""
        self.model = tf.keras.Sequential()

        if self.embedding == "use":
            self.model.add(InputLayer(input_shape=(self.embedding_dim,)))

            if self.model_type == "Dense":
                self.model.add(Dense(self.units, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
            else:
                self.model.add(Dense(self.units, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
                self.model.add(RepeatVector(self.max_length))

                if self.model_type == "RNN":
                    with tf.device("/CPU:0"):
                        self.model.add(SimpleRNN(self.units, return_sequences=False, dropout=0.3))
                elif self.model_type == "LSTM":
                    self.model.add(LSTM(self.units, return_sequences=False, dropout=0.3))
                elif self.model_type == "Bidirectional_LSTM":
                    self.model.add(Bidirectional(LSTM(self.units, return_sequences=False, dropout=0.3)))
                elif self.model_type == "GRU":
                    self.model.add(GRU(self.units, return_sequences=False, dropout=0.3))
                elif self.model_type == "Bidirectional_GRU":
                    self.model.add(Bidirectional(GRU(self.units, return_sequences=False, dropout=0.3)))

            self.model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

        else:
            if self.model_type == "Dense":
                raise ValueError("Dense model type is only supported with USE embeddings.")

            self.model.add(
                Embedding(
                    input_dim=vocab_size,
                    output_dim=self.embedding_dim,
                    weights=[embedding_matrix],
                    input_length=self.max_length,
                    trainable=True,
                )
            )

            if self.model_type == "RNN":
                with tf.device("/CPU:0"):
                    self.model.add(SimpleRNN(self.units, return_sequences=False, dropout=0.3))
            elif self.model_type == "LSTM":
                self.model.add(LSTM(self.units, return_sequences=False, dropout=0.3))
            elif self.model_type == "Bidirectional_LSTM":
                self.model.add(Bidirectional(LSTM(self.units, return_sequences=False, dropout=0.3)))
            elif self.model_type == "GRU":
                self.model.add(GRU(self.units, return_sequences=False, dropout=0.3))
            elif self.model_type == "Bidirectional_GRU":
                self.model.add(Bidirectional(GRU(self.units, return_sequences=False, dropout=0.3)))

            self.model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    # === METRICS ===
    def compute_metrics(self, y_true, y_pred, prefix: str = "") -> Dict[str, float]:
        """Compute and log evaluation metrics."""
        y_bin = (y_pred > 0.5).astype(int)

        negatives = np.sum(y_true == 0)
        accuracy = accuracy_score(y_true, y_bin)
        specificity = (
            np.sum((y_pred < 0.5) & (y_true == 0)) / negatives if negatives > 0 else float("nan")
        )
        precision = precision_score(y_true, y_bin)
        recall = recall_score(y_true, y_bin)
        f1 = f1_score(y_true, y_bin)
        auc = roc_auc_score(y_true, y_pred)

        mlflow.log_metric(f"{prefix}accuracy", accuracy)
        mlflow.log_metric(f"{prefix}specificity", specificity)
        mlflow.log_metric(f"{prefix}precision", precision)
        mlflow.log_metric(f"{prefix}recall", recall)
        mlflow.log_metric(f"{prefix}f1_score", f1)
        mlflow.log_metric(f"{prefix}roc_auc", auc)

        return {
            "accuracy": accuracy,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
        }

    # === TRAINING ===
    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train the model and evaluate on validation data."""
        if self.model is None:
            raise ValueError("The model has not been built. Call build_model first.")

        model_name = f"{self.model_type}_{self.embedding}"
        mlflow.log_param("embedding", self.embedding)
        mlflow.log_param("embedding_dim", self.embedding_dim)
        mlflow.log_param("units", self.units)
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("max_length", self.max_length)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]

        start_time = time.time()
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        )
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

        self._plot_learning_curves(history, model_name)

        y_val_pred = self.model.predict(X_val).flatten()
        val_metrics = self.compute_metrics(y_val, y_val_pred, prefix="val_")
        self._plot_roc_curve(y_val, y_val_pred, model_name)
        self._plot_confusion_matrix(y_val, y_val_pred, model_name)

        try:
            signature = infer_signature(X_val[:1], self.model.predict(X_val[:1]))
            self._log_model_with_registry(model_name, signature)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Error logging model or tokenizer to MLflow: {exc}")

        return val_metrics

    def _log_model_with_registry(self, model_name: str, signature) -> None:
        """Log the model to MLflow, falling back if the registry endpoint is unavailable."""

        def _log(registered_name: Optional[str]) -> None:
            mlflow.keras.log_model(
                self.model,
                artifact_path=model_name,
                registered_model_name=registered_name,
                signature=signature,
            )

        register_name = self.register_model_name
        try:
            _log(register_name)
        except MlflowException as exc:
            error_body = str(exc)
            error_code = getattr(exc, "error_code", None)
            status = getattr(exc, "status_code", None)
            if register_name and (
                "logged-models" in error_body
                or "404" in error_body
                or error_code in {"RESOURCE_DOES_NOT_EXIST", "ENDPOINT_NOT_FOUND"}
                or status == 404
            ):
                print(
                    "Model registry endpoint not available on the tracking server. "
                    "Continuing without registering the model."
                )
                _log(None)
            else:
                raise

        if self.tokenizer is not None:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
                pickle.dump(self.tokenizer, tmp_file)
                tokenizer_tmp_path = tmp_file.name
            mlflow.log_artifact(tokenizer_tmp_path, artifact_path=f"{model_name}/tokenizer")
            os.remove(tokenizer_tmp_path)

    # === VISUALISATION ===
    def _plot_learning_curves(self, history, model_name: str) -> None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"{model_name} - Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"{model_name} - Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            curve_path = tmp_file.name
        plt.savefig(curve_path)
        mlflow.log_artifact(curve_path, artifact_path="learning_curve")
        os.remove(curve_path)
        plt.close()

    def _plot_roc_curve(self, y_true, y_pred, model_name: str) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            roc_path = tmp_file.name
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path="roc_curve")
        os.remove(roc_path)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str) -> None:
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xticks([0.5, 1.5], ["Negative", "Positive"])
        plt.yticks([0.5, 1.5], ["Negative", "Positive"])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cm_path = tmp_file.name
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
        os.remove(cm_path)
        plt.close()

    # === FIT & TEST ===
    def fit(
        self,
        text: pd.Series,
        labels: pd.Series,
        test_size: float = 0.2,
        val_split: float = 0.2,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            text, labels, test_size=test_size, random_state=42, stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=42, stratify=y_temp
        )

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        print(f"Label distribution:\n{labels.value_counts()}")

        if self.embedding == "use":
            train_features = self._encode_with_use(X_train)
            val_features = self._encode_with_use(X_val)
            test_features = self._encode_with_use(X_test)

            self.build_model(vocab_size=None, embedding_matrix=None)
            val_metrics = self.train_and_evaluate(
                train_features, y_train, val_features, y_val, epochs, batch_size
            )
            y_test_pred = self.model.predict(test_features).flatten()
        else:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized for non-USE embeddings.")

            self.tokenizer.fit_on_texts(X_train)
            train_sequences = self.tokenizer.texts_to_sequences(X_train)
            train_padded_sequences = pad_sequences(
                train_sequences, maxlen=self.max_length, padding="post", truncating="post"
            )
            val_sequences = self.tokenizer.texts_to_sequences(X_val)
            val_padded_sequences = pad_sequences(
                val_sequences, maxlen=self.max_length, padding="post", truncating="post"
            )
            test_sequences = self.tokenizer.texts_to_sequences(X_test)
            test_padded_sequences = pad_sequences(
                test_sequences, maxlen=self.max_length, padding="post", truncating="post"
            )

            print(f"Sequence example (first 3):\n{train_sequences[:3]}")

            vocab_size = len(self.tokenizer.word_index) + 1
            embedding_matrix = self.build_embedding_matrix(X_train)
            self.build_model(vocab_size, embedding_matrix)

            val_metrics = self.train_and_evaluate(
                train_padded_sequences, y_train, val_padded_sequences, y_val, epochs, batch_size
            )
            y_test_pred = self.model.predict(test_padded_sequences).flatten()

        test_metrics = self.compute_metrics(y_test, y_test_pred, prefix="test_")
        print("Test metrics:", test_metrics)

        return {"val": val_metrics, "test": test_metrics}

    # === INFERENCE HELPERS ===
    def predict(self, texts: pd.Series):
        if self.model is None:
            raise ValueError("The model has not been built or trained. Call fit first.")

        if isinstance(texts, (list, tuple)):
            texts = pd.Series(texts)
        texts = texts.fillna("")

        if self.embedding == "use":
            features = self._encode_with_use(texts.reset_index(drop=True))
        else:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized for non-USE embeddings.")
            sequences = self.tokenizer.texts_to_sequences(texts)
            features = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")

        predictions = self.model.predict(features).flatten()
        return (predictions > 0.5).astype(int)

    def save(self, model_path: Optional[Path] = None) -> tuple[Path, Optional[Path]]:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if model_path is None:
            model_name = f"{self.model_type}_{self.embedding}"
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.keras"
            tokenizer_path = model_dir / "tokenizer.pkl"
        else:
            model_dir = model_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_path = model_dir / "tokenizer.pkl"

        self.model.save(model_path)
        if self.tokenizer is not None:
            with open(tokenizer_path, "wb") as f:
                pickle.dump(self.tokenizer, f)
        else:
            tokenizer_path = None

        return model_path, tokenizer_path

    @classmethod
    def load(cls, model_path, tokenizer_path, embedding, model_type="LSTM"):
        instance = cls(embedding=embedding, model_type=model_type)
        instance.model = load_model(model_path)
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                instance.tokenizer = pickle.load(f)
        elif instance.embedding != "use":
            raise ValueError("Tokenizer path must be provided for non-USE embeddings.")
        return instance


def flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {f"{split}_{metric}": value for split, values in metrics.items() for metric, value in values.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural sentiment models with MLflow tracking.")
    parser.add_argument("--data-path", default="data/processed_sample_tweets.csv", help="CSV dataset path.")
    parser.add_argument(
        "--text-column",
        default="advanced_processed_text_lem",
        help="Column name containing the preprocessed text.",
    )
    parser.add_argument("--target-column", default="target", help="Column name containing the target labels.")
    parser.add_argument(
        "--experiment-name",
        default="P7-Sentiments_Analysis_neural_network",
        help="MLflow experiment name.",
    )
    parser.add_argument("--run-name", default="Bidirectional_GRU_glove_training", help="MLflow run name.")
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLflow tracking URI override.")
    parser.add_argument("--embedding", default="glove", choices=["w2v", "fasttext", "glove", "use"])
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension.")
    parser.add_argument("--units", type=int, default=128, help="Number of units in recurrent layers.")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum sequence length.")
    parser.add_argument("--model-type", default="Bidirectional_GRU", help="Model type to train.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split proportion (of train set).")
    parser.add_argument(
        "--output-dir",
        default="backend/app/model",
        help="Directory where the exported model/tokenizer will be saved.",
    )
    parser.add_argument(
        "--model-filename",
        default="Bidirectional_GRU_glove_model.h5",
        help="Filename used for the exported Keras model.",
    )
    parser.add_argument(
        "--tokenizer-filename",
        default="tokenizer.pkl",
        help="Filename used for the exported tokenizer.",
    )
    parser.add_argument(
        "--register-model-name",
        default="Bidirectional_GRU_glove",
        help="Attempt to register the logged model under this name. Use an empty string to skip registration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(override=True)

    tracking_uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data_path)
    if args.text_column not in df or args.target_column not in df:
        raise KeyError(f"Columns '{args.text_column}' and '{args.target_column}' must exist in the dataset.")

    texts = df[args.text_column].astype(str).fillna("")
    labels = df[args.target_column].astype(int)

    register_name = args.register_model_name.strip() if args.register_model_name else None

    classifier = NeuralTweetClassifier(
        embedding=args.embedding,
        embedding_dim=args.embedding_dim,
        units=args.units,
        max_length=args.max_length,
        model_type=args.model_type,
        base_dir=".",
        register_model_name=register_name,
    )

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("text_column", args.text_column)
        mlflow.log_param("target_column", args.target_column)
        if register_name:
            mlflow.set_tag("register_model_name", register_name)

        metrics = classifier.fit(
            text=texts,
            labels=labels,
            test_size=args.test_size,
            val_split=args.val_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        mlflow.log_metrics(flatten_metrics(metrics))

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / args.model_filename
        tokenizer_path = output_dir / args.tokenizer_filename

        classifier.model.save(model_path, include_optimizer=False)
        if classifier.tokenizer is not None:
            with open(tokenizer_path, "wb") as f:
                pickle.dump(classifier.tokenizer, f)

        mlflow.log_artifact(model_path, artifact_path="exported_artifacts")
        if classifier.tokenizer is not None:
            mlflow.log_artifact(tokenizer_path, artifact_path="exported_artifacts")

        print("Training complete.")
        print("Validation metrics:", metrics["val"])
        print("Test metrics:", metrics["test"])
        print(f"Model exported to {model_path}")
        if classifier.tokenizer is not None:
            print(f"Tokenizer exported to {tokenizer_path}")


if __name__ == "__main__":
    main()
