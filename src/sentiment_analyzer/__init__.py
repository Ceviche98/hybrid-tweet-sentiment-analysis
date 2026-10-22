"""Hybrid Tweet Sentiment Analysis Package.

Combines tweet-aware preprocessing, multi-lexicon sentiment scoring (VADER, AFINN-165,
Hu & Liu opinion lexicon, Emoji UTF-8), n-gram feature protection, linear classifiers,
and SentenceTransformer dense semantic embeddings with DART-boosted LightGBM.
"""

from sentiment_analyzer.embeddings import HybridEmbeddingExtractor, SUPPORTED_ST_MODELS
from sentiment_analyzer.evaluation import (
    analyze_prediction_errors_with_weights,
    compute_weighted_metrics,
    confusion_matrix_heatmap,
    cross_validate,
    get_feature_weights_and_intercept,
)
from sentiment_analyzer.features import (
    build_vocabulary,
    calculate_feature_vector,
    extract_stylistic_meta_features,
    generate_ngrams,
    get_all_ngrams,
    global_feature_dict,
    reset_global_feature_dict,
    to_feature_vector,
    to_feature_vector_dictionary,
)
from sentiment_analyzer.lexicons import (
    META_FEATURE_NAMES_V1,
    META_FEATURE_NAMES_V2,
    extract_lexv1_features,
    extract_lexv2_features,
    is_lexicon_term,
    load_sentiment_lexicons,
)
from sentiment_analyzer.models import (
    EXPERIMENT_CONFIGS,
    HybridStackingClassifier,
    build_classifier,
    predict_labels_manual,
    predict_labels_nltk,
    train_baseline_nltk_classifier,
    train_classifier_manual,
)
from sentiment_analyzer.preprocessing import (
    BOOSTER_DICT,
    EXCLAM_BOOST,
    N_SCALAR,
    NEGATION_WORDS_SET,
    create_preprocessing_tools,
    load_and_split_raw_data,
    normalize_elongated_words,
    parse_data_line,
    pre_process_baseline,
    pre_process_pipeline,
    setup_nltk_resources,
)

__version__ = "1.0.0"

__all__ = [
    "BOOSTER_DICT",
    "EXCLAM_BOOST",
    "EXPERIMENT_CONFIGS",
    "HybridEmbeddingExtractor",
    "HybridStackingClassifier",
    "META_FEATURE_NAMES_V1",
    "META_FEATURE_NAMES_V2",
    "NEGATION_WORDS_SET",
    "N_SCALAR",
    "SUPPORTED_ST_MODELS",
    "analyze_prediction_errors_with_weights",
    "build_classifier",
    "build_vocabulary",
    "calculate_feature_vector",
    "compute_weighted_metrics",
    "confusion_matrix_heatmap",
    "create_preprocessing_tools",
    "cross_validate",
    "extract_lexv1_features",
    "extract_lexv2_features",
    "extract_stylistic_meta_features",
    "generate_ngrams",
    "get_all_ngrams",
    "get_feature_weights_and_intercept",
    "global_feature_dict",
    "is_lexicon_term",
    "load_and_split_raw_data",
    "load_sentiment_lexicons",
    "normalize_elongated_words",
    "parse_data_line",
    "pre_process_baseline",
    "pre_process_pipeline",
    "predict_labels_manual",
    "predict_labels_nltk",
    "reset_global_feature_dict",
    "setup_nltk_resources",
    "to_feature_vector",
    "to_feature_vector_dictionary",
    "train_baseline_nltk_classifier",
    "train_classifier_manual",
]
