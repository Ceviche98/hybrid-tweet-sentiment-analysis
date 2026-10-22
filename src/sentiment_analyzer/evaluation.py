"""10-fold cross-validation, weighted metrics, confusion matrix plotting, and FP/FN error analysis."""

import os
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from sentiment_analyzer.models import (
    predict_labels_manual,
    predict_labels_nltk,
    train_baseline_nltk_classifier,
    train_classifier_manual,
)


def compute_weighted_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, float]:
    """Compute weighted precision, recall, F1-score, and accuracy."""
    return {
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def cross_validate(
    dataset: List[Tuple[Dict[str, float], str]],
    folds: int = 10,
    vectorizer: Optional[DictVectorizer] = None,
    config: Optional[Dict[str, Any]] = None,
    return_train_scores: bool = False,
    stratified: bool = False,
    verbose: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """Perform k-fold cross-validation on `(feature_dict, label)` training data.

    Supports both the exact sequential fold splitting used in Question 2 (`stratified=False`)
    and stratified k-fold splitting (`stratified=True`). When `vectorizer` is `None`, uses
    the baseline NLTK `SklearnClassifier(LinearSVC())`; when `vectorizer` is provided,
    uses the fast vectorized `train_classifier_manual` pipeline.

    Args:
        dataset: List of `(feature_dict, label)` tuples.
        folds: Number of cross-validation folds (default 10).
        vectorizer: Optional fitted `DictVectorizer` for manual training.
        config: Optional model configuration dictionary.
        return_train_scores: If True, returns `(cv_results_train, cv_results_val)`.
        stratified: If True, uses `StratifiedKFold(n_splits=folds)`.
        verbose: If True, prints fold progress.

    Returns:
        Dictionary of averaged validation metrics (`precision`, `recall`, `f1`, `accuracy`),
        or `(train_metrics, val_metrics)` if `return_train_scores=True`.
    """
    results_val_per_fold: List[Dict[str, float]] = []
    results_train_per_fold: List[Dict[str, float]] = []

    fold_splits: List[Tuple[List[Tuple[Dict[str, float], str]], List[Tuple[Dict[str, float], str]]]] = []

    if stratified:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        labels = [d[1] for d in dataset]
        for train_idx, val_idx in skf.split(np.zeros(len(dataset)), labels):
            data_train = [dataset[idx] for idx in train_idx]
            data_val = [dataset[idx] for idx in val_idx]
            fold_splits.append((data_train, data_val))
    else:
        fold_size = int(len(dataset) / folds) + 1
        for i in range(0, len(dataset), fold_size):
            data_val = dataset[i : i + fold_size]
            data_train = dataset[:i] + dataset[i + fold_size :]
            if data_val and data_train:
                fold_splits.append((data_train, data_val))

    for num_fold, (data_train, data_validation) in enumerate(fold_splits, start=1):
        if verbose:
            print(f"Processing Fold {num_fold}/{len(fold_splits)}...")

        if vectorizer is None:
            model = train_baseline_nltk_classifier(data_train)
            if return_train_scores:
                tokens_train = list(map(itemgetter(0), data_train))
                true_label_train = list(map(itemgetter(1), data_train))
                pred_train = predict_labels_nltk(tokens_train, model)
                results_train_per_fold.append(compute_weighted_metrics(true_label_train, pred_train))

            tokens_val = list(map(itemgetter(0), data_validation))
            true_label_val = list(map(itemgetter(1), data_validation))
            pred_val = predict_labels_nltk(tokens_val, model)
            results_val_per_fold.append(compute_weighted_metrics(true_label_val, pred_val))
        else:
            model = train_classifier_manual(data_train, vectorizer, config)
            if return_train_scores:
                tokens_train = [d[0] for d in data_train]
                true_label_train = [d[1] for d in data_train]
                pred_train = predict_labels_manual(tokens_train, vectorizer, model)
                results_train_per_fold.append(compute_weighted_metrics(true_label_train, pred_train))

            tokens_val = [d[0] for d in data_validation]
            true_label_val = [d[1] for d in data_validation]
            pred_val = predict_labels_manual(tokens_val, vectorizer, model)
            results_val_per_fold.append(compute_weighted_metrics(true_label_val, pred_val))

    num_folds_actual = max(1, len(results_val_per_fold))
    cv_results_validation = {
        metric: sum(f[metric] for f in results_val_per_fold) / num_folds_actual
        for metric in ("precision", "recall", "f1", "accuracy")
    }

    if return_train_scores:
        cv_results_training = {
            metric: sum(f[metric] for f in results_train_per_fold) / num_folds_actual
            for metric in ("precision", "recall", "f1", "accuracy")
        }
        return cv_results_training, cv_results_validation

    return cv_results_validation


def confusion_matrix_heatmap(
    y_test: Sequence[str],
    preds: Sequence[str],
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> np.ndarray:
    """Plot and optionally save a labeled confusion matrix heatmap (Question 3)."""
    cm = metrics.confusion_matrix(y_test, preds, labels=labels)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Confusion matrix of the classifier")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="w")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    b, t = plt.ylim()
    b += 0.1
    t -= 0.1
    plt.ylim(b, t)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return cm


def get_feature_weights_and_intercept(
    classifier: Any,
    vectorizer: Optional[DictVectorizer] = None,
) -> Tuple[Dict[str, float], float]:
    """Extract linear feature weights `{token: weight}` and scalar intercept from a classifier.

    Supports both NLTK `SklearnClassifier` objects and manual `(LinearSVC/LogisticRegression, DictVectorizer)`.
    """
    try:
        if hasattr(classifier, "_clf") and hasattr(classifier, "_vectorizer"):
            pipeline = classifier._clf
            linear_svc = pipeline.named_steps["svc"]
            vec = classifier._vectorizer
            coefficients = linear_svc.coef_[0]
            feature_names = vec.get_feature_names_out()
            intercept_value = float(linear_svc.intercept_[0])
            # Determine if NLTK label encoder inverted sign (0='negative', 1='positive')
            if hasattr(classifier, "_encoder") and hasattr(classifier._encoder, "classes_"):
                classes = list(classifier._encoder.classes_)
                if classes == ["positive", "negative"]:
                    coefficients = -coefficients
                    intercept_value = -intercept_value
            return dict(zip(feature_names, map(float, coefficients))), intercept_value

        if vectorizer is not None and hasattr(classifier, "coef_"):
            coefficients = classifier.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            intercept_value = float(classifier.intercept_[0])
            if hasattr(classifier, "classes_") and list(classifier.classes_) == ["positive", "negative"]:
                coefficients = -coefficients
                intercept_value = -intercept_value
            return dict(zip(feature_names, map(float, coefficients))), intercept_value
    except Exception:
        pass
    return {}, 0.0


def analyze_prediction_errors_with_weights(
    classifier: Any,
    validation_with_predictions: List[Tuple[Tuple[Union[List[str], Dict[str, float]], str], str]],
    n: int = 3,
    output_dir: Optional[Union[str, Path]] = None,
    vectorizer: Optional[DictVectorizer] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Inspect False Positive and False Negative predictions with token-level linear weights.

    Calculates each token's learned weight, the sum of token weights, the model intercept (bias),
    and the resulting margin score (`final_score = token_sum + intercept`), saving all FP and FN
    instances to `false_positives.txt` and `false_negatives.txt`.

    Args:
        classifier: Trained `SklearnClassifier` or linear model.
        validation_with_predictions: List of `((tokens_or_dict, true_label), predicted_label)`.
        n: Number of top FP/FN examples to inspect in detail.
        output_dir: Directory where `false_negatives.txt` and `false_positives.txt` are written.
        vectorizer: Optional `DictVectorizer` if `classifier` is a raw scikit-learn linear model.
        verbose: Whether to print detailed analysis to stdout.

    Returns:
        Dictionary containing `intercept`, `fn_count`, `fp_count`, `fn_examples`, `fp_examples`,
        `fn_file_path`, and `fp_file_path`.
    """
    feature_weights, intercept = get_feature_weights_and_intercept(classifier, vectorizer=vectorizer)
    if not feature_weights:
        return {
            "intercept": 0.0,
            "fn_count": 0,
            "fp_count": 0,
            "fn_examples": [],
            "fp_examples": [],
        }

    out_base = Path(output_dir) if output_dir is not None else Path(".")
    out_base.mkdir(parents=True, exist_ok=True)
    fn_file_path = out_base / "false_negatives.txt"
    fp_file_path = out_base / "false_positives.txt"

    if fn_file_path.exists():
        os.remove(fn_file_path)
    if fp_file_path.exists():
        os.remove(fp_file_path)

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL ERROR ANALYSIS")
        print(f"Model Intercept (Bias): {intercept:+.4f}")
        print("=" * 60)

    fn_count = 0
    fp_count = 0
    fn_examples: List[Dict[str, Any]] = []
    fp_examples: List[Dict[str, Any]] = []

    def _extract_tokens_and_counts(
        rep: Union[List[str], Dict[str, float]]
    ) -> Tuple[List[str], float, List[Tuple[str, float]]]:
        if isinstance(rep, dict):
            tokens_seq: List[str] = []
            for tok, cnt in rep.items():
                tokens_seq.extend([tok] * max(1, int(round(cnt))))
        else:
            tokens_seq = list(rep)

        token_weights_list = [feature_weights.get(tok, 0.0) for tok in tokens_seq]
        total_sum = float(sum(token_weights_list))
        token_weights = [(tok, float(feature_weights.get(tok, 0.0))) for tok in tokens_seq]
        sorted_weights = sorted(token_weights, key=lambda item: item[1], reverse=True)
        return tokens_seq, total_sum, sorted_weights

    with open(fn_file_path, "w", encoding="utf-8") as fn_file:
        for original_data, predicted in validation_with_predictions:
            original_rep, true_label = original_data
            if true_label == "positive" and predicted == "negative":
                fn_count += 1
                tokens_seq, total_token_sum, sorted_token_weights = _extract_tokens_and_counts(
                    original_rep
                )
                error_text = " ".join(tokens_seq)
                final_score = total_token_sum + intercept
                fn_file.write(
                    f"Score: {final_score:+.4f} | True: {true_label} | Predicted: {predicted} | Text: {error_text}\n"
                )
                if fn_count <= n:
                    example_record = {
                        "text": error_text,
                        "token_sum": total_token_sum,
                        "intercept": intercept,
                        "final_score": final_score,
                        "token_weights": sorted_token_weights,
                    }
                    fn_examples.append(example_record)
                    if verbose:
                        print(f"\n--- FN Example {fn_count}: {error_text} ---")
                        print(
                            f"  Token Sum: {total_token_sum:+.4f} | Intercept: {intercept:+.4f} | Final Score: {final_score:+.4f}"
                        )
                        for tok, w in sorted_token_weights:
                            print(f"    {tok:<20} (weight: {w:+.4f})")

    with open(fp_file_path, "w", encoding="utf-8") as fp_file:
        for original_data, predicted in validation_with_predictions:
            original_rep, true_label = original_data
            if true_label == "negative" and predicted == "positive":
                fp_count += 1
                tokens_seq, total_token_sum, sorted_token_weights = _extract_tokens_and_counts(
                    original_rep
                )
                error_text = " ".join(tokens_seq)
                final_score = total_token_sum + intercept
                fp_file.write(
                    f"Score: {final_score:+.4f} | True: {true_label} | Predicted: {predicted} | Text: {error_text}\n"
                )
                if fp_count <= n:
                    example_record = {
                        "text": error_text,
                        "token_sum": total_token_sum,
                        "intercept": intercept,
                        "final_score": final_score,
                        "token_weights": sorted_token_weights,
                    }
                    fp_examples.append(example_record)
                    if verbose:
                        print(f"\n--- FP Example {fp_count}: {error_text} ---")
                        print(
                            f"  Token Sum: {total_token_sum:+.4f} | Intercept: {intercept:+.4f} | Final Score: {final_score:+.4f}"
                        )
                        for tok, w in sorted_token_weights:
                            print(f"    {tok:<20} (weight: {w:+.4f})")

    return {
        "intercept": intercept,
        "fn_count": fn_count,
        "fp_count": fp_count,
        "fn_examples": fn_examples,
        "fp_examples": fp_examples,
        "fn_file_path": str(fn_file_path),
        "fp_file_path": str(fp_file_path),
    }
