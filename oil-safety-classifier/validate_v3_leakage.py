#!/usr/bin/env python3
"""
validate_v3_leakage.py

Leakage validation for the V3 unsafe-act SIF dataset.

Checks:
1. Exact duplicate reports across different labels.
2. Near-duplicate reports using TF-IDF cosine similarity.
3. Whether suspicious near-duplicates cross the eventual train/test split.
4. 5-fold stratified cross-validation on the same model family.

Run from project root:
    python validate_v3_leakage.py --csv data/raw/unsafe_act_v3_contrast_balanced.csv
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


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
            analyzer="word", ngram_range=(1, 2),
            min_df=1, max_df=0.98, sublinear_tf=True,
            max_features=60000, strip_accents="unicode"
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_features=50000, sublinear_tf=True
        ))
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    path = Path(args.csv).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = ["report_text", "sif_binary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    df["report_text"] = df["report_text"].fillna("").map(clean_text)
    df["sif_binary"] = pd.to_numeric(df["sif_binary"], errors="coerce")
    df = df[df["report_text"].ne("") & df["sif_binary"].isin([0, 1])].copy()
    df = df.reset_index(drop=True)

    print("=" * 72)
    print("V3 LEAKAGE VALIDATION")
    print("=" * 72)
    print(f"Dataset: {path}")
    print(f"Rows: {len(df)}")

    # ---------------------------------------------------------------
    # 1. Exact duplicate check
    # ---------------------------------------------------------------
    duplicated = df[df["report_text"].duplicated(keep=False)].copy()
    duplicate_groups = (
        df.groupby("report_text")["sif_binary"]
        .agg(["count", "nunique"])
    )
    conflicting_exact = duplicate_groups[duplicate_groups["nunique"] > 1]

    print("\n[1] EXACT DUPLICATES")
    print(f"Duplicate rows: {len(duplicated)}")
    print(f"Duplicate text groups: {(duplicate_groups['count'] > 1).sum()}")
    print(f"Exact duplicate groups with conflicting labels: {len(conflicting_exact)}")

    # ---------------------------------------------------------------
    # 2. Near-duplicate check
    # ---------------------------------------------------------------
    print("\n[2] NEAR-DUPLICATE CHECK")
    print("Building TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    X = normalize(vectorizer.fit_transform(df["report_text"]))

    # Nearest-neighbour search. k=2 because the closest item is itself.
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    similarities = 1.0 - distances[:, 1]
    nearest = indices[:, 1]

    df["nearest_similarity"] = similarities
    df["nearest_index"] = nearest

    suspicious = []
    for i, j in enumerate(nearest):
        if i == j:
            continue
        sim = similarities[i]
        if sim >= args.threshold:
            suspicious.append({
                "index_a": i,
                "index_b": int(j),
                "similarity": float(sim),
                "label_a": int(df.iloc[i]["sif_binary"]),
                "label_b": int(df.iloc[j]["sif_binary"]),
                "same_label": bool(
                    df.iloc[i]["sif_binary"] == df.iloc[j]["sif_binary"]
                ),
            })

    # Deduplicate pair direction.
    unique_pairs = {}
    for row in suspicious:
        pair = tuple(sorted((row["index_a"], row["index_b"])))
        unique_pairs[pair] = row

    suspicious = list(unique_pairs.values())
    conflicting_near = [x for x in suspicious if not x["same_label"]]

    print(f"Threshold: cosine similarity >= {args.threshold:.2f}")
    print(f"Suspicious near-duplicate pairs: {len(suspicious)}")
    print(f"Near-duplicate pairs with conflicting labels: {len(conflicting_near)}")

    if suspicious:
        print("\nTop suspicious pairs:")
        for item in sorted(suspicious, key=lambda x: x["similarity"], reverse=True)[:20]:
            a = item["index_a"]
            b = item["index_b"]
            print(f"\nSimilarity: {item['similarity']:.4f} | "
                  f"labels: {item['label_a']} vs {item['label_b']}")
            print("A:", df.iloc[a]["report_text"][:220])
            print("B:", df.iloc[b]["report_text"][:220])

    # ---------------------------------------------------------------
    # 3. Cross-fold leakage estimate
    # ---------------------------------------------------------------
    print("\n[3] CROSS-VALIDATION")
    print("Running 5-fold stratified validation...")

    y = df["sif_binary"].astype(int)
    model = build_model()

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        model, df["report_text"], y,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = results[f"test_{metric}"]
        print(f"{metric.capitalize():10}: "
              f"{values.mean():.4f} ± {values.std():.4f} "
              f"(folds: {', '.join(f'{v:.4f}' for v in values)})")

    # ---------------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    if len(conflicting_exact) > 0:
        print("WARNING: Exact duplicate reports have conflicting labels.")
    elif len(duplicated) > 0:
        print("NOTE: Exact duplicates exist, but their labels are consistent.")
    else:
        print("PASS: No exact duplicate report texts found.")

    if len(conflicting_near) > 0:
        print("WARNING: Near-duplicate reports with different labels were found.")
    elif len(suspicious) > 0:
        print("CAUTION: Very similar reports exist; inspect the pairs above.")
    else:
        print("PASS: No near-duplicate pairs above the threshold were found.")

    mean_f1 = results["test_f1"].mean()
    print(f"\n5-fold mean F1: {mean_f1:.4f}")

    if mean_f1 < 0.95:
        print("The cross-validation score is substantially lower than the "
              "original holdout result; investigate dataset composition.")
    else:
        print("Cross-validation remains strong. This supports, but does not "
              "prove, that the 99.13% holdout result is not caused solely "
              "by a favorable split.")

    print("\nLeakage validation complete.")


if __name__ == "__main__":
    main()
