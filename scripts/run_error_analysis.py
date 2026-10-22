#!/usr/bin/env python3
"""CLI entrypoint for Question 3 Confusion Matrix & Weight-Based Error Analysis."""

import argparse
import sys
from operator import itemgetter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sentiment_analyzer import (  # noqa: E402
    analyze_prediction_errors_with_weights,
    confusion_matrix_heatmap,
    load_and_split_raw_data,
    pre_process_baseline,
    predict_labels_nltk,
    to_feature_vector_dictionary,
    train_baseline_nltk_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run first-fold confusion matrix and token weight FP/FN error inspection."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "sentiment-dataset.tsv",
        help="Path to tab-separated sentiment dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory to save false_positives.txt, false_negatives.txt, and confusion matrix.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of False Positive and False Negative examples to print in full detail.",
    )
    args = parser.parse_args()

    raw_train, _ = load_and_split_raw_data(args.dataset, percentage=0.8)
    train_data = [
        (to_feature_vector_dictionary(pre_process_baseline(text)), label)
        for text, label in raw_train
    ]

    fold_size = int(len(train_data) / 10) + 1
    data_validation = train_data[:fold_size]
    data_train = train_data[fold_size:]

    print(f"Training baseline LinearSVC on {len(data_train)} samples (Folds 2-10)...")
    model = train_baseline_nltk_classifier(data_train)

    tokens_validation = list(map(itemgetter(0), data_validation))
    true_label_validation = list(map(itemgetter(1), data_validation))
    predicted_labels = predict_labels_nltk(tokens_validation, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = args.output_dir / "confusion_matrix_fold1.png"
    cm = confusion_matrix_heatmap(
        true_label_validation,
        predicted_labels,
        labels=["positive", "negative"],
        save_path=cm_path,
        show=False,
    )
    print(f"Confusion Matrix (Fold 1) saved to {cm_path}:")
    print(f"  True Positives (pos->pos): {cm[0, 0]}")
    print(f"  False Negatives (pos->neg): {cm[0, 1]}")
    print(f"  False Positives (neg->pos): {cm[1, 0]}")
    print(f"  True Negatives (neg->neg): {cm[1, 1]}")

    # Reconstruct token lists for weight-based inspection
    raw_val_tokens = [
        (pre_process_baseline(text), label) for text, label in raw_train[:fold_size]
    ]
    validation_with_predictions = list(zip(raw_val_tokens, predicted_labels))

    analyze_prediction_errors_with_weights(
        classifier=model,
        validation_with_predictions=validation_with_predictions,
        n=args.top_n,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
