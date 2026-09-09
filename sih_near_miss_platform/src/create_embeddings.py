import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    ML_READY_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL_NAME,
    create_project_directories
)


def main():
    create_project_directories()

    if not ML_READY_FILE.exists():
        raise FileNotFoundError(
            f"ML-ready file not found: {ML_READY_FILE}. "
            "Run data_preparation.py first."
        )

    df = pd.read_csv(ML_READY_FILE)

    if "model_text" not in df.columns:
        raise ValueError(
            "Column 'model_text' is missing from the ML-ready dataset."
        )

    texts = (
        df["model_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if not texts:
        raise ValueError(
            "No report text was found for embedding creation."
        )

    print(f"Reports loaded: {len(texts)}")
    print(f"Loading NLP model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Creating Sentence-BERT embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print("\n--- Embedding Creation Complete ---")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved embeddings: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()