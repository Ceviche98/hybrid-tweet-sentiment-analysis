"""Unit tests for the hybrid tweet sentiment analysis pipeline."""

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sentiment_analyzer import (  # noqa: E402
    HybridEmbeddingExtractor,
    META_FEATURE_NAMES_V1,
    META_FEATURE_NAMES_V2,
    analyze_prediction_errors_with_weights,
    build_classifier,
    build_vocabulary,
    calculate_feature_vector,
    compute_weighted_metrics,
    confusion_matrix_heatmap,
    create_preprocessing_tools,
    cross_validate,
    extract_lexv1_features,
    extract_lexv2_features,
    extract_stylistic_meta_features,
    generate_ngrams,
    get_all_ngrams,
    global_feature_dict,
    is_lexicon_term,
    load_sentiment_lexicons,
    normalize_elongated_words,
    pre_process_baseline,
    pre_process_pipeline,
    reset_global_feature_dict,
    to_feature_vector_dictionary,
    train_baseline_nltk_classifier,
)


@pytest.fixture(scope="module")
def lexicons():
    lex_dir = REPO_ROOT / "assets" / "lexicons"
    return load_sentiment_lexicons(lex_dir)


@pytest.fixture(scope="module")
def tools(lexicons):
    return create_preprocessing_tools(lexicons)


def test_lexicons_loaded(lexicons):
    assert len(lexicons["bing_liu_positive"]) == 2006
    assert len(lexicons["bing_liu_negative"]) == 4783
    assert len(lexicons["afinn_scores"]) > 3000
    assert len(lexicons["vader_scores"]) > 7000
    assert len(lexicons["emoji_descriptions"]) > 3000


def test_baseline_bow_vectorization():
    reset_global_feature_dict()
    tokens = pre_process_baseline("Hello Hello I am Oscar")
    vec = to_feature_vector_dictionary(tokens)
    assert vec == {"Hello": 2.0, "I": 1.0, "am": 1.0, "Oscar": 1.0}
    assert "Hello" in global_feature_dict
    assert len(global_feature_dict) == 4


def test_tweet_preprocessing_option1_and_option2(tools):
    tweet = (
        "RT @CGBlasi: Very impressed w/ Joco! :) :( "
        "I am NOT happy... https://t.co/abc #Fight"
    )
    cfg_v1 = {
        "feature_set": "option_1",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "keep_word",
    }
    tokens_v1 = pre_process_pipeline(tweet, tools, cfg_v1)
    assert "@cgblasi" not in tokens_v1
    assert "rt" not in tokens_v1
    assert "https" not in " ".join(tokens_v1)
    assert "__emoticon_pos__" in tokens_v1
    assert "__emoticon_neg__" in tokens_v1
    # Negation word 'not' must be preserved even when stopwords are removed
    assert "not" in tokens_v1
    # Hashtag word preserved without '#'
    assert "fight" in tokens_v1
    assert "!" in tokens_v1

    cfg_v2 = {
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "handle_hashtags": "remove",
    }
    tokens_v2 = pre_process_pipeline(tweet, tools, cfg_v2)
    assert "fight" not in tokens_v2
    assert "not" in tokens_v2


def test_elongated_word_normalization():
    assert normalize_elongated_words("sooooo goooood!!!") == "soo good!!"


def test_ngram_generation():
    tokens = ["not", "very", "good"]
    assert generate_ngrams(tokens, 2) == ["not_very", "very_good"]
    assert generate_ngrams(tokens, 3) == ["not_very_good"]
    all_ng = get_all_ngrams(tokens, {"ngram_range": (1, 3)})
    assert "not" in all_ng
    assert "not_very" in all_ng
    assert "not_very_good" in all_ng


def test_lexv1_and_lexv2_scoring(tools):
    tokens_pos = ["extremely", "good", "!", "__emoticon_pos__"]
    ngrams_pos = get_all_ngrams(tokens_pos, {"ngram_range": (1, 3)})
    v1_pos = extract_lexv1_features(tokens_pos, ngrams_pos, tools)
    assert set(v1_pos.keys()) == set(META_FEATURE_NAMES_V1)
    assert v1_pos["__LEX_POS_COUNT__"] >= 1.0
    assert v1_pos["__META_POS_EMO_COUNT__"] == 1.0
    assert v1_pos["__META_EXCLAM_COUNT__"] == 1.0

    tokens_negated = ["not", "very", "good", "!"]
    ngrams_negated = get_all_ngrams(tokens_negated, {"ngram_range": (1, 3)})
    v1_neg = extract_lexv1_features(tokens_negated, ngrams_negated, tools)
    assert v1_neg["__META_NEGATED_POS_COUNT__"] >= 1.0

    v2_pos = extract_lexv2_features(["very", "good", "!"], tools)
    v2_neg = extract_lexv2_features(["not", "good", "!"], tools)
    assert set(v2_pos.keys()) == set(META_FEATURE_NAMES_V2)
    assert v2_pos["__LEX_SCORE_VADER__"] > 0.0
    assert v2_pos["__LEX_SCORE_AFINN__"] > 0.0
    # Negation scalar (-0.74) inverts positive polarity into negative score
    assert v2_neg["__LEX_SCORE_VADER__"] < 0.0
    assert v2_neg["__LEX_SCORE_AFINN__"] < 0.0
    assert v2_neg["__META_NEGATED_POS_COUNT__"] == 1.0


def test_vocabulary_pruning_and_lexicon_protection(tools):
    clean_train = [
        (["common", "word"], "positive"),
        (["common", "word"], "positive"),
        (["common", "rare_non_lex"], "negative"),
        (["not_good", "awesome"], "negative"),  # 'awesome' and 'not_good' are lexicon terms appearing once
    ]
    cfg = {
        "feature_set": "option_2",
        "ngram_range": (1, 2),
        "min_df": 2,
        "protect_lexicon_features": True,
    }
    vocab, total_docs = build_vocabulary(clean_train, tools, cfg)
    assert total_docs == 4
    assert "common" in vocab
    assert "rare_non_lex" not in vocab
    # Protected by Hu & Liu lexicon even though df == 1 < min_df (2)
    assert is_lexicon_term("awesome", tools)
    assert "awesome" in vocab
    assert "not_good" in vocab


def test_stylistic_meta_features():
    feats = extract_stylistic_meta_features("WOW AMAZING!!?")
    assert feats["__STYLE_CAPS_RATIO__"] == 1.0
    assert feats["__STYLE_EXCLAM_RAW__"] == 2.0
    assert feats["__STYLE_QMARK_RAW__"] == 1.0


def test_cross_validation_and_error_analysis(tmp_path, tools):
    synthetic_raw = [
        ("I love this amazing wonderful film! :)", "positive"),
        ("Super happy fantastic day great joy!", "positive"),
        ("Best experience ever awesome service!", "positive"),
        ("Really good wonderful time loved it!", "positive"),
        ("Awesome great lovely brilliant work!", "positive"),
        ("I hate this terrible awful horrible movie :(", "negative"),
        ("Worst disaster bad disgusting failure!", "negative"),
        ("Awfulhorrible sad angry frustrating waste!", "negative"),
        ("Terrible service bad rude disappointing!", "negative"),
        ("Hate worst awful nasty painful experience!", "negative"),
    ] * 4  # 40 samples

    cfg = {
        "feature_set": "option_2",
        "lemmatize": True,
        "remove_stopwords": True,
        "ngram_range": (1, 2),
        "min_df": 1,
        "use_tfidf": False,
        "model_type": "lr",
        "lr_C": 1.0,
    }
    clean_data = [(pre_process_pipeline(t, tools, cfg), lbl) for t, lbl in synthetic_raw]
    vocab, n_docs = build_vocabulary(clean_data, tools, cfg)
    dataset = [
        (calculate_feature_vector(toks, tools, cfg, vocab=vocab, total_docs=n_docs), lbl)
        for toks, lbl in clean_data
    ]

    vec = DictVectorizer()
    vec.fit([d[0] for d in dataset])

    cv_train, cv_val = cross_validate(
        dataset,
        folds=5,
        vectorizer=vec,
        config=cfg,
        return_train_scores=True,
        stratified=True,
    )
    assert 0.8 <= cv_val["f1"] <= 1.0
    assert 0.8 <= cv_train["f1"] <= 1.0

    # Test NLTK baseline classifier and error analysis
    nltk_model = train_baseline_nltk_classifier(dataset)
    val_with_preds = [
        ((["good", "bad"], "positive"), "negative"),  # FN
        ((["awful", "love"], "negative"), "positive"),  # FP
    ]
    report = analyze_prediction_errors_with_weights(
        nltk_model,
        val_with_preds,
        n=2,
        output_dir=tmp_path,
        verbose=False,
    )
    assert report["fn_count"] == 1
    assert report["fp_count"] == 1
    assert Path(report["fn_file_path"]).exists()
    assert Path(report["fp_file_path"]).exists()

    cm_file = tmp_path / "cm.png"
    cm = confusion_matrix_heatmap(
        ["positive", "positive", "negative", "negative"],
        ["positive", "negative", "positive", "negative"],
        labels=["positive", "negative"],
        save_path=cm_file,
        show=False,
    )
    assert cm.shape == (2, 2)
    assert cm_file.exists()


def test_hybrid_embedding_extractor_lexv2_matrix(tools):
    extractor = HybridEmbeddingExtractor(tools=tools)
    mat = extractor.extract_lexv2_matrix(["I love this!", "I do not like this terrible thing!"], fit_vectorizer=True)
    assert mat.shape == (2, len(META_FEATURE_NAMES_V2))
    assert mat.dtype == np.float32


def test_model_builders():
    for mtype in ("svc", "lr", "lgbm", "lgbm_st", "rf", "mlp"):
        clf = build_classifier({"model_type": mtype})
        assert clf is not None
