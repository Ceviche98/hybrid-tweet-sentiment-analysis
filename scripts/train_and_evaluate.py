#!/usr/bin/env python3
"""CLI entrypoint for training and evaluating tweet sentiment classification models."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sentiment_analyzer import (  # noqa: E402
    EXPERIMENT_CONFIGS,
    HybridEmbeddingExtractor,
    build_classifier,
    build_vocabulary,
    calculate_feature_vector,
    compute_weighted_metrics,
    create_preprocessing_tools,
    cross_validate,
    load_and_split_raw_data,
    load_sentiment_lexicons,
    pre_process_baseline,
    pre_process_pipeline,
    predict_labels_manual,
    predict_labels_nltk,
    to_feature_vector_dictionary,
    train_baseline_nltk_classifier,
    train_classifier_manual,
)


def run_experiment(
    dataset_path: Path,
    lexicon_dir: Path,
    config: Dict[str, Any],
    folds: int = 10,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Execute a full 10-fold CV + held-out test evaluation for a given experiment config."""
    raw_train, raw_test = load_and_split_raw_data(dataset_path, percentage=0.8)
    if max_samples > 0:
        raw_train = raw_train[: int(max_samples * 0.8)]
        raw_test = raw_test[: max(10, int(max_samples * 0.2))]

    print(f"Loaded {len(raw_train)} training tweets and {len(raw_test)} test tweets.")
    print(f"Running experiment: {config.get('name', 'Custom Experiment')}")

    if config.get("baseline_mode", False):
        train_data = [
            (to_feature_vector_dictionary(pre_process_baseline(text)), label)
            for text, label in raw_train
        ]
        test_data = [
            (to_feature_vector_dictionary(pre_process_baseline(text)), label)
            for text, label in raw_test
        ]
        cv_val = cross_validate(train_data, folds=folds, verbose=True)
        model = train_baseline_nltk_classifier(train_data)
        test_preds = predict_labels_nltk([x[0] for x in test_data], model)
        test_metrics = compute_weighted_metrics([x[1] for x in test_data], test_preds)
        print(f"\n[CV Validation] F1: {cv_val['f1']:.4f} | Acc: {cv_val['accuracy']:.4f}")
        print(f"[Held-Out Test] F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f}")
        return {"cv_val": cv_val, "test": test_metrics}

    lexicons = load_sentiment_lexicons(lexicon_dir)
    tools = create_preprocessing_tools(lexicons)

    # SentenceTransformer + LexV2 + LightGBM branch
    if "st_model_name" in config:
        st_name = config["st_model_name"]
        extractor = HybridEmbeddingExtractor(
            model_name=st_name,
            tools=tools,
            preproc_config=config,
        )
        X_train = extractor.fit_transform([t for t, _ in raw_train], show_progress_bar=True)
        y_train = np.array([lbl for _, lbl in raw_train])
        X_test = extractor.transform([t for t, _ in raw_test], show_progress_bar=True)
        y_test = np.array([lbl for _, lbl in raw_test])

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        val_f1s = []
        train_f1s = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
            clf = build_classifier(config)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            tr_preds = clf.predict(X_train[tr_idx])
            val_preds = clf.predict(X_train[val_idx])
            tr_m = compute_weighted_metrics(y_train[tr_idx], tr_preds)
            val_m = compute_weighted_metrics(y_train[val_idx], val_preds)
            train_f1s.append(tr_m["f1"])
            val_f1s.append(val_m["f1"])
            print(f"  Fold {fold_idx}/{folds} -> Train F1: {tr_m['f1']:.4f} | Val F1: {val_m['f1']:.4f}")

        final_clf = build_classifier(config)
        final_clf.fit(X_train, y_train)
        test_preds = final_clf.predict(X_test)
        test_metrics = compute_weighted_metrics(y_test, test_preds)
        print(f"\n[CV Avg] Train F1: {np.mean(train_f1s):.4f} | Val F1: {np.mean(val_f1s):.4f}")
        print(f"[Held-Out Test] F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f}")
        return {
            "cv_train_f1": float(np.mean(train_f1s)),
            "cv_val_f1": float(np.mean(val_f1s)),
            "test": test_metrics,
        }

    # Traditional / LexV1 / LexV2 pipeline
    clean_train = [
        (pre_process_pipeline(text, tools, config), label) for text, label in raw_train
    ]
    vocab, total_train_docs = build_vocabulary(clean_train, tools, config)
    print(f"Vocabulary size after pruning: {len(vocab):,} features")

    train_data_final = [
        (calculate_feature_vector(tokens, tools, config, vocab=vocab, total_docs=total_train_docs), label)
        for tokens, label in clean_train
    ]

    vectorizer = DictVectorizer()
    vectorizer.fit([{term: 1.0} for term in vocab.keys()])

    cv_train, cv_val = cross_validate(
        train_data_final,
        folds=folds,
        vectorizer=vectorizer,
        config=config,
        return_train_scores=True,
        verbose=True,
    )

    final_model = train_classifier_manual(train_data_final, vectorizer, config)

    test_vectors = []
    test_labels = []
    for text, label in raw_test:
        clean_tokens = pre_process_pipeline(text, tools, config)
        vec = calculate_feature_vector(
            clean_tokens, tools, config, vocab=vocab, total_docs=total_train_docs
        )
        test_vectors.append(vec)
        test_labels.append(label)

    test_preds = predict_labels_manual(test_vectors, vectorizer, final_model)
    test_metrics = compute_weighted_metrics(test_labels, test_preds)

    print(f"\n[CV Train]      F1: {cv_train['f1']:.4f} | Acc: {cv_train['accuracy']:.4f}")
    print(f"[CV Validation] F1: {cv_val['f1']:.4f} | Acc: {cv_val['accuracy']:.4f}")
    print(f"[Held-Out Test] F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f}")
    return {"cv_train": cv_train, "cv_val": cv_val, "test": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate hybrid tweet sentiment analysis models."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "sentiment-dataset.tsv",
        help="Path to tab-separated sentiment dataset.",
    )
    parser.add_argument(
        "--lexicon-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "lexicons",
        help="Path to sentiment lexicons directory.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        default="lr_lexv2_best",
        help="Preset experiment configuration from Table I.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total samples for rapid smoke testing (0 = full dataset).",
    )
    args = parser.parse_args()

    config = dict(EXPERIMENT_CONFIGS[args.preset])
    run_experiment(
        dataset_path=args.dataset,
        lexicon_dir=args.lexicon_dir,
        config=config,
        folds=args.folds,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
