import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 128
DEFAULT_MODEL_DIR = "model"


def load_imdb_dataset(num_words: int = VOCAB_SIZE, maxlen: int = MAX_SEQUENCE_LENGTH) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load the IMDb dataset and prepare padded sequences."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    x_train = pad_sequences(x_train, maxlen=maxlen, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=maxlen, padding="post", truncating="post")
    return (x_train, y_train), (x_test, y_test)


def get_word_index(num_words: int = VOCAB_SIZE) -> Dict[str, int]:
    """Build a vocabulary mapping compatible with the IMDb dataset."""
    raw_index = imdb.get_word_index()
    index = {word: (rank + 3) for word, rank in raw_index.items()}
    index["<PAD>"] = 0
    index["<START>"] = 1
    index["<UNK>"] = 2
    index["<UNUSED>"] = 3
    return {word: idx for word, idx in index.items() if idx < num_words}


def text_to_sequence(text: str, word_index: Dict[str, int], maxlen: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    """Convert raw review text into a padded integer sequence."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    sequence = [word_index.get(token, word_index.get("<UNK>", 2)) for token in tokens]
    return pad_sequences([sequence], maxlen=maxlen, padding="post", truncating="post")[0]


def build_model(vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM, maxlen: int = MAX_SEQUENCE_LENGTH) -> tf.keras.Model:
    """Build and compile the sentiment classification model."""
    model = models.Sequential(
        [
            layers.Input(shape=(maxlen,), dtype="int32"),
            layers.Embedding(vocab_size, embedding_dim),
            layers.Conv1D(128, 5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model_dir: Path, epochs: int = 6, batch_size: int = 512) -> None:
    """Train the model and save it to disk."""
    (x_train, y_train), (x_test, y_test) = load_imdb_dataset()
    model = build_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[callback],
        verbose=2,
    )

    # Save model. Keras 3 requires an extension for the native Keras format
    # or use `export` to write a SavedModel directory. If the provided
    # `model_dir` has a file extension we save to that path, otherwise we
    # export a SavedModel into the directory.
    if model_dir.suffix in (".keras", ".h5"):
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))
        print(f"Saved trained model to {model_dir}")
    else:
        # Export SavedModel to a directory (recommended for serving/tflite)
        model.export(str(model_dir))
        print(f"Exported trained model to {model_dir}")


def evaluate_model(model_dir: Path) -> None:
    """Evaluate a saved model on the IMDb test dataset."""
    model = tf.keras.models.load_model(model_dir)
    (_, _), (x_test, y_test) = load_imdb_dataset()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


def predict_review(text: str, model_dir: Path) -> str:
    """Predict sentiment for a single review string."""
    model = tf.keras.models.load_model(model_dir)
    word_index = get_word_index()
    sequence = text_to_sequence(text, word_index)
    score = float(model.predict(np.array([sequence]), verbose=0)[0, 0])
    label = "positive" if score >= 0.5 else "negative"
    return f"{label} ({score:.3f})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMDb Sentiment Analysis")
    parser.add_argument("--train", action="store_true", help="Train a new sentiment model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model on test data")
    parser.add_argument("--predict", type=str, help="Predict sentiment for a raw review")
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR), help="Path to save or load the model")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any([args.train, args.evaluate, args.predict]):
        args.train = True
        args.evaluate = True

    if args.train:
        train_model(args.model_dir, epochs=args.epochs, batch_size=args.batch_size)

    if args.evaluate:
        evaluate_model(args.model_dir)

    if args.predict:
        print(predict_review(args.predict, args.model_dir))


if __name__ == "__main__":
    main()
