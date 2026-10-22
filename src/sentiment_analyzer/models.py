"""Unified model builders and hyperparameter configurations for sentiment classification."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nltk.classify import SklearnClassifier
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def train_baseline_nltk_classifier(
    data: List[Tuple[Dict[str, float], str]],
) -> SklearnClassifier:
    """Train the baseline NLTK SklearnClassifier wrapping LinearSVC (Question 1-3)."""
    pipeline = Pipeline([("svc", LinearSVC())])
    return SklearnClassifier(pipeline).train(data)


def predict_labels_nltk(
    samples: List[Dict[str, float]],
    classifier: SklearnClassifier,
) -> List[str]:
    """Predict sentiment labels using a trained NLTK SklearnClassifier."""
    return list(classifier.classify_many(samples))


def build_classifier(config: Optional[Dict[str, Any]] = None) -> Any:
    """Build a scikit-learn or LightGBM classifier according to configuration parameters.

    Supports:
    - `svc` / `linearsvc`: Linear Support Vector Classifier (`C`, `penalty`, `class_weight`, `max_iter`)
    - `lr` / `logistic_regression`: Logistic Regression (`C`, `penalty`, `class_weight`, `solver`)
    - `lgbm` / `lightgbm`: LightGBM (`gbdt` or `dart` boosting, `n_estimators`, `max_depth`,
      `learning_rate`, `feature_fraction`, `lambda_l1`, `lambda_l2`, `is_unbalance`)
    - `rf` / `random_forest`: Random Forest Classifier
    - `mlp`: Multi-Layer Perceptron Classifier
    """
    if config is None:
        config = {}

    model_type = config.get("model_type", "svc").lower()

    if model_type in ("svc", "linearsvc"):
        svm_c = config.get("svm_C", 1.0)
        svm_weight = config.get("svm_class_weight", None)
        svm_iter = config.get("svm_class_iter", 1000)
        svm_penalty = config.get("svm_class_penalty", "l2")
        svm_loss = config.get("svm_class_loss", "squared_hinge")
        dual = config.get("svm_dual", False if svm_penalty == "l1" else "auto")
        return LinearSVC(
            dual=dual,
            C=svm_c,
            class_weight=svm_weight,
            penalty=svm_penalty,
            max_iter=svm_iter,
            loss=svm_loss,
            random_state=config.get("random_state", 42),
        )

    if model_type in ("lr", "logistic_regression"):
        lr_c = config.get("lr_C", 0.25)
        lr_penalty = config.get("lr_penalty", "l2")
        lr_weight = config.get("lr_class_weight", "balanced")
        lr_iter = config.get("lr_max_iter", 3000)
        return LogisticRegression(
            solver="liblinear",
            C=lr_c,
            penalty=lr_penalty,
            class_weight=lr_weight,
            max_iter=lr_iter,
            random_state=config.get("random_state", 42),
        )

    if model_type in ("lgbm", "lightgbm"):
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=config.get("lgbm_n_estimators", 300),
                learning_rate=config.get("lgbm_learning_rate", 0.1),
                max_depth=config.get("lgbm_max_depth", 7),
                random_state=config.get("random_state", 42),
            )

        params: Dict[str, Any] = {
            "objective": "binary",
            "n_estimators": config.get("lgbm_n_estimators", 900),
            "learning_rate": config.get("lgbm_learning_rate", 0.1),
            "max_depth": config.get("lgbm_max_depth", 7),
            "is_unbalance": config.get("lgbm_is_unbalance", True),
            "boosting_type": config.get("lgbm_boosting_type", "gbdt"),
            "random_state": config.get("random_state", 42),
            "n_jobs": config.get("n_jobs", -1),
            "verbose": -1,
        }
        if "lgbm_feature_fraction" in config:
            params["feature_fraction"] = config["lgbm_feature_fraction"]
        if "lgbm_subsample" in config:
            params["subsample"] = config["lgbm_subsample"]
        if "lgbm_lambda_l1" in config:
            params["lambda_l1"] = config["lgbm_lambda_l1"]
        if "lgbm_lambda_l2" in config:
            params["lambda_l2"] = config["lgbm_lambda_l2"]
        if "lgbm_min_child_samples" in config:
            params["min_child_samples"] = config["lgbm_min_child_samples"]
        return LGBMClassifier(**params)

    if model_type in ("lgbm_st", "lightgbm_st"):
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=config.get("lgbm_n_estimators", 500),
                learning_rate=config.get("lgbm_learning_rate", 0.2),
                max_depth=config.get("lgbm_max_depth", 8),
                l2_regularization=config.get("lgbm_lambda_l2", 20.0),
                random_state=config.get("random_state", 42),
            )

        return LGBMClassifier(
            objective="binary",
            n_estimators=config.get("lgbm_n_estimators", 1500),
            learning_rate=config.get("lgbm_learning_rate", 0.2),
            max_depth=config.get("lgbm_max_depth", 8),
            is_unbalance=config.get("lgbm_is_unbalance", True),
            random_state=config.get("random_state", 42),
            lambda_l1=config.get("lgbm_lambda_l1", 20),
            lambda_l2=config.get("lgbm_lambda_l2", 20),
            subsample=config.get("lgbm_subsample", 0.6),
            min_child_samples=config.get("lgbm_min_child_samples", 70),
            boosting_type=config.get("lgbm_boosting_type", "dart"),
            n_jobs=config.get("n_jobs", -1),
            verbose=-1,
        )

    if model_type in ("rf", "random_forest"):
        return RandomForestClassifier(
            n_estimators=config.get("rf_n_estimators", 300),
            max_depth=config.get("rf_max_depth", None),
            class_weight=config.get("rf_class_weight", "balanced"),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", -1),
        )

    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.get("mlp_hidden_layers", (128, 64)),
            max_iter=config.get("mlp_max_iter", 200),
            random_state=config.get("random_state", 42),
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def train_classifier_manual(
    data: List[Tuple[Dict[str, float], str]],
    vectorizer: DictVectorizer,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train a classifier on `(feature_dict, label)` pairs using a fitted `DictVectorizer`."""
    X_dicts = [d[0] for d in data]
    y_labels = [d[1] for d in data]
    X_matrix = vectorizer.transform(X_dicts)

    model = build_classifier(config)
    model.fit(X_matrix, y_labels)
    return model


def predict_labels_manual(
    samples: List[Dict[str, float]],
    vectorizer: DictVectorizer,
    model: Any,
) -> List[str]:
    """Predict labels for a list of feature dictionaries using a fitted vectorizer and model."""
    X_matrix = vectorizer.transform(samples)
    return list(model.predict(X_matrix))


class HybridStackingClassifier:
    """Two-level stacking ensemble: Level-1 (LinearSVC + LogisticRegression) -> Level-2 (LightGBM).

    Level-1 models generate decision function margin scores over the full n-gram + LexV2
    feature space, which are horizontally concatenated with the 11 LexV2 meta-features to
    train the Level-2 DART-boosted LightGBM meta-learner.
    """

    def __init__(self, random_state: int = 42, lgbm_n_estimators: int = 1500) -> None:
        self.random_state = random_state
        self.lgbm_n_estimators = lgbm_n_estimators
        self.l1_svc = LinearSVC(
            dual=False,
            C=0.2,
            class_weight="balanced",
            penalty="l1",
            max_iter=3000,
            loss="squared_hinge",
            random_state=random_state,
        )
        self.l1_lr = LogisticRegression(
            solver="liblinear",
            C=0.25,
            penalty="l2",
            class_weight="balanced",
            max_iter=3000,
            random_state=random_state,
        )
        self.l2_lgbm: Any = None

    def _build_l2_model(self) -> Any:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=self.lgbm_n_estimators,
            learning_rate=0.2,
            max_depth=8,
            is_unbalance=True,
            random_state=self.random_state,
            lambda_l1=20,
            lambda_l2=20,
            subsample=0.8,
            min_child_samples=70,
            boosting_type="dart",
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X_l1: Any, X_meta: Any, y: np.ndarray) -> "HybridStackingClassifier":
        """Fit Level-1 SVC and LR models and Level-2 LightGBM meta-classifier."""
        self.l1_svc.fit(X_l1, y)
        self.l1_lr.fit(X_l1, y)

        scores_svc = self.l1_svc.decision_function(X_l1).reshape(-1, 1)
        scores_lr = self.l1_lr.decision_function(X_l1).reshape(-1, 1)

        X_l2 = hstack([X_meta, scores_svc, scores_lr]).tocsr()
        self.l2_lgbm = self._build_l2_model()
        self.l2_lgbm.fit(X_l2, y)
        return self

    def predict(self, X_l1: Any, X_meta: Any) -> np.ndarray:
        """Predict sentiment labels using the fitted two-stage stack."""
        scores_svc = self.l1_svc.decision_function(X_l1).reshape(-1, 1)
        scores_lr = self.l1_lr.decision_function(X_l1).reshape(-1, 1)
        X_l2 = hstack([X_meta, scores_svc, scores_lr]).tocsr()
        return np.asarray(self.l2_lgbm.predict(X_l2))


EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_bow": {
        "name": "Baseline (Raw Split + TF BoW + LinearSVC)",
        "baseline_mode": True,
        "model_type": "svc",
    },
    "lemma": {
        "name": "Lemma",
        "lemmatize": True,
        "remove_stopwords": False,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 1),
        "min_df": 1,
        "use_tfidf": False,
        "model_type": "svc",
    },
    "lemma_stop": {
        "name": "Lemma + Stop",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 1),
        "min_df": 1,
        "use_tfidf": False,
        "model_type": "svc",
    },
    "stem": {
        "name": "Stem",
        "stem": True,
        "remove_stopwords": False,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 1),
        "min_df": 1,
        "use_tfidf": False,
        "model_type": "svc",
    },
    "stem_stop": {
        "name": "Stem + Stop",
        "stem": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 1),
        "min_df": 1,
        "use_tfidf": False,
        "model_type": "svc",
    },
    "svc_lexv1_best": {
        "name": "Lemma + Stop + N[3] + MIN[3] + LexV1 + SVC(C:0.2,cw:balanced,pen:l1,iter=3000)",
        "feature_set": "option_1",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 3),
        "min_df": 3,
        "use_tfidf": False,
        "protect_lexicon_features": True,
        "model_type": "svc",
        "svm_C": 0.2,
        "svm_class_weight": "balanced",
        "svm_class_iter": 3000,
        "svm_class_penalty": "l1",
        "svm_class_loss": "squared_hinge",
    },
    "svc_lexv2_best": {
        "name": "Lemma + Stop + N[3] + MIN[3] + LexV2 + SVC(C:0.2,cw:balanced,pen:l1,iter=3000)",
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 3),
        "min_df": 3,
        "use_tfidf": False,
        "protect_lexicon_features": True,
        "model_type": "svc",
        "svm_C": 0.2,
        "svm_class_weight": "balanced",
        "svm_class_iter": 3000,
        "svm_class_penalty": "l1",
        "svm_class_loss": "squared_hinge",
    },
    "lr_lexv2_best": {
        "name": "Lemma + Stop + N[3] + MIN[3] + LexV2 + LR(C:0.25,cw:balanced,pen:l2,iter=3000)",
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 3),
        "min_df": 3,
        "use_tfidf": False,
        "protect_lexicon_features": True,
        "model_type": "lr",
        "lr_C": 0.25,
        "lr_penalty": "l2",
        "lr_class_weight": "balanced",
        "lr_max_iter": 3000,
    },
    "lgbm_lexv2": {
        "name": "Lemma + Stop + N[3] + MIN[3] + LexV2 + LGBM(n_est:900,m_depth:7,is_unbal:True,ff:0.3,lr:0.1)",
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "ngram_range": (1, 3),
        "min_df": 3,
        "use_tfidf": False,
        "protect_lexicon_features": True,
        "model_type": "lgbm",
        "lgbm_n_estimators": 900,
        "lgbm_max_depth": 7,
        "lgbm_learning_rate": 0.1,
        "lgbm_feature_fraction": 0.3,
        "lgbm_is_unbalance": True,
    },
    "st_mpnet_lgbm": {
        "name": "ST: all-mpnet-base-v2 + LexV2 -> LGBM",
        "st_model_name": "all-mpnet-base-v2",
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
        "model_type": "lgbm_st",
    },
}
