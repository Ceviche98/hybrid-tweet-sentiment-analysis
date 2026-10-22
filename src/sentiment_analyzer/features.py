"""Term-Frequency Bag-of-Words, N-gram generators, and tweet meta-feature extractors."""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from sentiment_analyzer.lexicons import (
    META_FEATURE_NAMES_V1,
    META_FEATURE_NAMES_V2,
    extract_lexv1_features,
    extract_lexv2_features,
    is_lexicon_term,
)

# Global feature dictionary maintained for baseline compatibility (Question 1)
global_feature_dict: Dict[str, int] = {}


def reset_global_feature_dict() -> None:
    """Clear the global feature dictionary."""
    global global_feature_dict
    global_feature_dict.clear()


def to_feature_vector_dictionary(
    tokens: List[str],
    vocab_tracker: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Transform a list of tokens into a Term-Frequency (TF) Bag-of-Words dictionary.

    Args:
        tokens: List of preprocessed string tokens.
        vocab_tracker: Optional dictionary tracking global vocabulary occurrences.
            Updates `global_feature_dict` by default.

    Returns:
        Dictionary mapping each unique token in the document to its term frequency.
    """
    token_counter = Counter(tokens)
    feature_vector: Dict[str, float] = {k: float(v) for k, v in token_counter.items()}

    target_vocab = vocab_tracker if vocab_tracker is not None else global_feature_dict
    for token in token_counter.keys():
        if token not in target_vocab:
            target_vocab[token] = 1
        else:
            target_vocab[token] += 1

    return feature_vector


# Alias matching the original coursework template name
to_feature_vector = to_feature_vector_dictionary


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate contiguous n-grams joined by '_' (e.g., `['not', 'good']` -> `['not_good']`)."""
    if n <= 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def get_all_ngrams(tokens: List[str], config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate all n-grams within `config['ngram_range']` (default `(1, 1)`)."""
    if config is None:
        config = {}
    min_n, max_n = config.get("ngram_range", (1, 1))
    all_ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            all_ngrams.extend(tokens)
        else:
            all_ngrams.extend(generate_ngrams(tokens, n))
    return all_ngrams


def extract_stylistic_meta_features(raw_text: str) -> Dict[str, float]:
    """Extract raw character/stylistic tweet meta-features (caps ratio, punctuation, length)."""
    alpha_chars = [c for c in raw_text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    caps_ratio = len(upper_chars) / len(alpha_chars) if alpha_chars else 0.0
    words = raw_text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0

    return {
        "__STYLE_CAPS_RATIO__": float(caps_ratio),
        "__STYLE_CHAR_LEN__": float(len(raw_text)),
        "__STYLE_WORD_COUNT__": float(len(words)),
        "__STYLE_AVG_WORD_LEN__": float(avg_word_len),
        "__STYLE_EXCLAM_RAW__": float(raw_text.count("!")),
        "__STYLE_QMARK_RAW__": float(raw_text.count("?")),
    }


def build_vocabulary(
    clean_train_data: List[Tuple[List[str], str]],
    tools: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[Dict[str, int], int]:
    """Build document-frequency (DF) vocabulary with `min_df` pruning and lexicon protection.

    Args:
        clean_train_data: List of `(clean_tokens, label)` tuples from training split.
        tools: Preprocessing and lexicon tools dictionary.
        config: Experiment configuration dictionary.

    Returns:
        Tuple of `(vocab_df_dict, total_train_docs)`.
    """
    df_counter: Counter = Counter()
    for tokens, _ in clean_train_data:
        all_ngrams = get_all_ngrams(tokens, config)
        df_counter.update(set(all_ngrams))

    vocab: Dict[str, int] = dict(df_counter)
    total_train_docs = len(clean_train_data)

    feature_set = config.get("feature_set", None)
    use_lex_or_meta = (
        feature_set in ("option_1", "option_2")
        or config.get("use_lexicon_features", False)
        or config.get("use_meta_features", False)
    )

    if use_lex_or_meta:
        meta_names = (
            META_FEATURE_NAMES_V2 if feature_set == "option_2" else META_FEATURE_NAMES_V1
        )
        for meta_name in meta_names:
            vocab[meta_name] = total_train_docs

    min_df = config.get("min_df", 1)
    protect_lexicon = config.get("protect_lexicon_features", False)

    if min_df > 1 or protect_lexicon:
        pruned_vocab: Dict[str, int] = {}
        for term, freq in vocab.items():
            if freq >= min_df:
                pruned_vocab[term] = freq
            elif protect_lexicon and is_lexicon_term(term, tools):
                pruned_vocab[term] = freq
        vocab = pruned_vocab

    return vocab, total_train_docs


def calculate_feature_vector(
    clean_tokens: List[str],
    tools: Dict[str, Any],
    config: Dict[str, Any],
    vocab: Optional[Dict[str, int]] = None,
    total_docs: int = 1,
) -> Dict[str, float]:
    """Compute the combined n-gram (TF or TF-IDF) + LexV1/LexV2 feature dictionary for a tweet.

    Args:
        clean_tokens: Preprocessed tokens for a single tweet.
        tools: Preprocessing and lexicon tools dictionary.
        config: Experiment configuration dictionary.
        vocab: Fitted document-frequency vocabulary dictionary (defaults to `global_feature_dict`).
        total_docs: Total number of training documents (for IDF computation).

    Returns:
        Dictionary mapping feature names to float weights.
    """
    active_vocab = vocab if vocab is not None else global_feature_dict
    feature_vector: Dict[str, float] = {}

    use_tfidf = config.get("use_tfidf", False)
    all_ngrams = get_all_ngrams(clean_tokens, config)
    tf_counts = Counter(all_ngrams)

    for term, tf in tf_counts.items():
        if active_vocab and term not in active_vocab:
            continue
        if use_tfidf:
            df = active_vocab.get(term, 1)
            idf = math.log(max(1, total_docs) / (df + 1))
            feature_vector[term] = float(tf) * idf
        else:
            feature_vector[term] = float(tf)

    feature_set_type = config.get("feature_set", None)
    use_lex_v1 = config.get("use_lexicon_features", False) or config.get("use_meta_features", False)

    if feature_set_type == "option_2":
        feature_vector.update(extract_lexv2_features(clean_tokens, tools))
    elif feature_set_type == "option_1" or use_lex_v1:
        feature_vector.update(extract_lexv1_features(clean_tokens, all_ngrams, tools))

    return feature_vector
