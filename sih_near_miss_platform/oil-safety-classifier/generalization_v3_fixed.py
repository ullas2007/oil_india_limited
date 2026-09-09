#!/usr/bin/env python3
"""
generalization_v3.py

Stricter generalization test for the V3 unsafe-act SIF classifier.

It groups highly similar reports together BEFORE splitting. Therefore,
near-duplicate/template variants cannot land in both train and test folds.

Run from project root:
    python generalization_v3.py --csv "data/raw/unsafe_act_v3_contrast_balanced.csv"
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


RANDOM_STATE = 42
N_SPLITS = 5


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[^a-z0-9\s%/\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_model():
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=60000,
            strip_accents="unicode",
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])


def make_similarity_groups(texts, threshold):
    """
    Build connected components where reports are linked when cosine
    similarity is >= threshold. The connected component becomes the
    group, preventing similar template variants from crossing folds.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = normalize(vectorizer.fit_transform(texts))

    # At 0.90 similarity, only retrieve neighbours close enough to be
    # candidates. n_neighbors is deliberately larger than 2 because
    # several reports can belong to the same template family.
    n_neighbors = min(30, len(texts))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    parent = np.arange(len(texts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    links = 0

    for i in range(len(texts)):
        for pos in range(1, indices.shape[1]):
            j = int(indices[i, pos])
            similarity = 1.0 - float(distances[i, pos])
            if similarity >= threshold:
                union(i, j)
                links += 1
            else:
                # Neighbours are ordered by distance; once below the
                # threshold, later neighbours will also be below it.
                break

    roots = {}
    groups = np.zeros(len(texts), dtype=int)

    next_group = 0
    for i in range(len(texts)):
        root = find(i)
        if root not in roots:
            roots[root] = next_group
            next_group += 1
        groups[i] = roots[root]

    return groups, links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for grouping similar reports.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required = ["report_text", "sif_binary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    df["report_text"] = df["report_text"].fillna("").map(clean_text)
    df["sif_binary"] = pd.to_numeric(df["sif_binary"], errors="coerce")

    df = df[
        df["report_text"].ne("") &
        df["sif_binary"].isin([0, 1])
    ].copy()

    df = df.drop_duplicates(subset=["report_text"]).reset_index(drop=True)

    print("=" * 72)
    print("V3 STRICT GENERALIZATION VALIDATION")
    print("=" * 72)
    print(f"Dataset: {csv_path}")
    print(f"Usable reports: {len(df)}")
    print(f"Grouping threshold: {args.threshold:.2f}")

    print("\nCreating similarity groups...")
    groups, links = make_similarity_groups(
        df["report_text"].tolist(),
        args.threshold,
    )

    df["similarity_group"] = groups

    n_groups = df["similarity_group"].nunique()
    group_sizes = df["similarity_group"].value_counts()

    print(f"Similarity links: {links}")
    print(f"Similarity groups: {n_groups}")
    print(f"Largest group: {group_sizes.max()} reports")
    print(f"Singleton groups: {(group_sizes == 1).sum()}")

    # Verify groups are label-consistent.
    group_label_counts = (
        df.groupby("similarity_group")["sif_binary"]
        .nunique()
    )
    conflicting_groups = int((group_label_counts > 1).sum())

    print(f"Groups containing both labels: {conflicting_groups}")

    if conflicting_groups:
        print(
            "\nWARNING: Some similarity groups contain both labels. "
            "This may indicate ambiguous or problematic generated data."
        )

    X = df["report_text"]
    y = df["sif_binary"].astype(int)
    g = df["similarity_group"]

    print("\nRunning 5-fold StratifiedGroupKFold...")
    print(
        "Near-duplicate/template groups are kept together, so a "
        "similar report cannot appear in both train and test."
    )

    model = build_model()

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
    }

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = cross_validate(
        model,
        X,
        y,
        groups=g,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    print("\n" + "=" * 72)
    print("STRICT GENERALIZATION RESULTS")
    print("=" * 72)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = results[f"test_{metric}"]
        print(
            f"{metric.capitalize():10}: "
            f"{values.mean():.4f} ± {values.std():.4f} "
            f"(folds: {', '.join(f'{v:.4f}' for v in values)})"
        )

    mean_accuracy = results["test_accuracy"].mean()
    mean_precision = results["test_precision"].mean()
    mean_recall = results["test_recall"].mean()
    mean_f1 = results["test_f1"].mean()

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    print(f"Original random-split accuracy: 99.13%")
    print(f"Original random-split F1:       99.15%")
    print(f"Strict grouped accuracy:        {mean_accuracy:.2%}")
    print(f"Strict grouped F1:              {mean_f1:.2%}")

    if mean_f1 >= 0.95:
        print(
            "\nSTRONG: V3 still generalizes well when similar/template "
            "reports are kept out of the opposite fold."
        )
    elif mean_f1 >= 0.85:
        print(
            "\nMODERATE: V3 generalizes reasonably, but performance "
            "drops under the stricter test."
        )
    else:
        print(
            "\nWEAK: V3 performance drops substantially under the "
            "near-duplicate-aware test. Do not claim the random-split "
            "99% result as real-world performance."
        )

    print(
        "\nNote: This is a stronger dataset-internal generalization test. "
        "The best final validation is still a genuinely independent "
        "real-world dataset collected from different sources/time periods."
    )


if __name__ == "__main__":
    main()
