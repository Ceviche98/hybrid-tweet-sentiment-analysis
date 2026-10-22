"""Multi-lexicon sentiment feature extraction (LexV1 and LexV2)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from sentiment_analyzer.preprocessing import (
    BOOSTER_DICT,
    EXCLAM_BOOST,
    N_SCALAR,
    NEGATION_WORDS_SET,
)

META_FEATURE_NAMES_V1: List[str] = [
    "__LEX_POS_COUNT__",
    "__LEX_NEG_COUNT__",
    "__META_TOKEN_COUNT__",
    "__META_POS_EMO_COUNT__",
    "__META_NEG_EMO_COUNT__",
    "__META_EXCLAM_COUNT__",
    "__META_QMARK_COUNT__",
    "__META_NEGATED_POS_COUNT__",
    "__META_NEGATED_NEG_COUNT__",
]

META_FEATURE_NAMES_V2: List[str] = [
    "__LEX_POS_COUNT__",
    "__LEX_NEG_COUNT__",
    "__LEX_SCORE_AFINN__",
    "__LEX_SCORE_VADER__",
    "__LEX_SCORE_BINGLIU_NET__",
    "__LEX_SCORE_AVERAGE__",
    "__META_TOKEN_COUNT__",
    "__META_EXCLAM_COUNT__",
    "__META_QMARK_COUNT__",
    "__META_NEGATED_POS_COUNT__",
    "__META_NEGATED_NEG_COUNT__",
]


def get_default_lexicon_dir() -> Path:
    """Resolve the default `assets/lexicons` directory relative to the package root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "assets" / "lexicons"


def load_sentiment_lexicons(
    lexicon_dir: Optional[Union[str, Path]] = None,
    bing_pos: Optional[Union[str, Path]] = None,
    bing_neg: Optional[Union[str, Path]] = None,
    afinn_words: Optional[Union[str, Path]] = None,
    afinn_emo: Optional[Union[str, Path]] = None,
    vader_lex: Optional[Union[str, Path]] = None,
    emoji_lex: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load Hu & Liu (Bing Liu), AFINN-165, AFINN-emoticon-8, VADER, and Emoji lexicons.

    Args:
        lexicon_dir: Directory containing lexicon `.txt` files. Defaults to `assets/lexicons/`.
        bing_pos: Optional explicit path to `positive-words.txt`.
        bing_neg: Optional explicit path to `negative-words.txt`.
        afinn_words: Optional explicit path to `AFINN-en-165.txt`.
        afinn_emo: Optional explicit path to `AFINN-emoticon-8.txt`.
        vader_lex: Optional explicit path to `vader_lexicon.txt`.
        emoji_lex: Optional explicit path to `emoji_utf8_lexicon.txt`.

    Returns:
        Dictionary containing `bing_liu_positive`, `bing_liu_negative`, `afinn_scores`,
        `vader_scores`, and `emoji_descriptions`.
    """
    base_dir = Path(lexicon_dir) if lexicon_dir is not None else get_default_lexicon_dir()

    pos_path = Path(bing_pos) if bing_pos else base_dir / "positive-words.txt"
    neg_path = Path(bing_neg) if bing_neg else base_dir / "negative-words.txt"
    afinn_w_path = Path(afinn_words) if afinn_words else base_dir / "AFINN-en-165.txt"
    afinn_e_path = Path(afinn_emo) if afinn_emo else base_dir / "AFINN-emoticon-8.txt"
    vader_path = Path(vader_lex) if vader_lex else base_dir / "vader_lexicon.txt"
    emoji_path = Path(emoji_lex) if emoji_lex else base_dir / "emoji_utf8_lexicon.txt"

    lexicons: Dict[str, Any] = {
        "bing_liu_positive": set(),
        "bing_liu_negative": set(),
        "afinn_scores": {},
        "vader_scores": {},
        "emoji_descriptions": {},
    }

    if pos_path.exists():
        with open(pos_path, "r", encoding="ISO-8859-1") as f:
            lexicons["bing_liu_positive"] = {
                line.strip() for line in f if line.strip() and not line.startswith(";")
            }

    if neg_path.exists():
        with open(neg_path, "r", encoding="ISO-8859-1") as f:
            lexicons["bing_liu_negative"] = {
                line.strip() for line in f if line.strip() and not line.startswith(";")
            }

    afinn_scores: Dict[str, float] = {}
    for af_path in (afinn_w_path, afinn_e_path):
        if af_path.exists():
            with open(af_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        afinn_scores[parts[0]] = float(parts[1])
    lexicons["afinn_scores"] = afinn_scores

    vader_scores: Dict[str, float] = {}
    if vader_path.exists():
        with open(vader_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    vader_scores[parts[0]] = float(parts[1])
    lexicons["vader_scores"] = vader_scores

    emoji_descriptions: Dict[str, str] = {}
    if emoji_path.exists():
        with open(emoji_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    emoji_descriptions[parts[0]] = parts[1]
    lexicons["emoji_descriptions"] = emoji_descriptions

    return lexicons


def is_lexicon_term(term: str, lexicons: Dict[str, Any]) -> bool:
    """Check whether a unigram or underscore-joined n-gram contains a Hu & Liu lexicon word.

    Used during vocabulary pruning (`protect_lexicon_features=True`) so rare
    sentiment-bearing words and n-grams are exempted from minimum document frequency filtering.
    """
    pos_set: Set[str] = lexicons.get("bing_liu_positive", set())
    neg_set: Set[str] = lexicons.get("bing_liu_negative", set())
    if not pos_set and "lexicons" in lexicons:
        pos_set = lexicons["lexicons"].get("positive", set())
        neg_set = lexicons["lexicons"].get("negative", set())

    if term in pos_set or term in neg_set:
        return True
    if "_" in term:
        for part in term.split("_"):
            if part in pos_set or part in neg_set:
                return True
    return False


def extract_lexv1_features(
    clean_tokens: List[str],
    all_ngrams: List[str],
    tools: Dict[str, Any],
) -> Dict[str, float]:
    """Extract LexV1 count-based lexicon and tweet meta-features.

    Features computed:
    - `__LEX_POS_COUNT__`, `__LEX_NEG_COUNT__`: Unigram hits in Hu & Liu opinion lexicon.
    - `__META_TOKEN_COUNT__`: Total cleaned tokens.
    - `__META_POS_EMO_COUNT__`, `__META_NEG_EMO_COUNT__`: Grouped ASCII emoticon counts.
    - `__META_EXCLAM_COUNT__`, `__META_QMARK_COUNT__`: Punctuation emphasis counts.
    - `__META_NEGATED_POS_COUNT__`, `__META_NEGATED_NEG_COUNT__`: N-grams starting with a
      negation word and ending with a positive/negative lexicon word (e.g., `not_good`, `not_very_good`).
    """
    bing_pos_lex: Set[str] = tools.get("bing_liu_positive", set())
    bing_neg_lex: Set[str] = tools.get("bing_liu_negative", set())
    if not bing_pos_lex and "lexicons" in tools:
        bing_pos_lex = tools["lexicons"].get("positive", set())
        bing_neg_lex = tools["lexicons"].get("negative", set())

    negation_words: Set[str] = tools.get("negation_words", NEGATION_WORDS_SET)

    bing_pos_count = 0
    bing_neg_count = 0
    negated_pos_count = 0
    negated_neg_count = 0

    for token in clean_tokens:
        if token in bing_pos_lex:
            bing_pos_count += 1
        if token in bing_neg_lex:
            bing_neg_count += 1

    for ngram in all_ngrams:
        if "_" not in ngram:
            continue
        parts = ngram.split("_")
        if parts[0] in negation_words:
            last_word = parts[-1]
            if last_word in bing_pos_lex:
                negated_pos_count += 1
            elif last_word in bing_neg_lex:
                negated_neg_count += 1

    return {
        "__LEX_POS_COUNT__": float(bing_pos_count),
        "__LEX_NEG_COUNT__": float(bing_neg_count),
        "__META_TOKEN_COUNT__": float(len(clean_tokens)),
        "__META_POS_EMO_COUNT__": float(clean_tokens.count("__emoticon_pos__")),
        "__META_NEG_EMO_COUNT__": float(clean_tokens.count("__emoticon_neg__")),
        "__META_EXCLAM_COUNT__": float(clean_tokens.count("!")),
        "__META_QMARK_COUNT__": float(clean_tokens.count("?")),
        "__META_NEGATED_POS_COUNT__": float(negated_pos_count),
        "__META_NEGATED_NEG_COUNT__": float(negated_neg_count),
    }


def extract_lexv2_features(
    clean_tokens: List[str],
    tools: Dict[str, Any],
) -> Dict[str, float]:
    """Extract LexV2 continuous multi-lexicon sentiment features with VADER/AFINN heuristics.

    Incorporates:
    - Hu & Liu positive/negative counts and net polarity (`__LEX_SCORE_BINGLIU_NET__`).
    - Continuous AFINN-165 + AFINN-emoticon-8 and VADER valence scores.
    - 3-token lookbehind negation inversion (`vader_neg_scalar = -0.74`).
    - 2-token lookbehind booster/dampener scaling (`+/- 0.293`).
    - 2-token lookahead exclamation amplification (`+0.292` for VADER, `+1.0` for AFINN).
    - Composite ensemble lexicon average (`__LEX_SCORE_AVERAGE__`).
    """
    bing_pos_lex: Set[str] = tools.get("bing_liu_positive", set())
    bing_neg_lex: Set[str] = tools.get("bing_liu_negative", set())
    if not bing_pos_lex and "lexicons" in tools:
        bing_pos_lex = tools["lexicons"].get("positive", set())
        bing_neg_lex = tools["lexicons"].get("negative", set())

    afinn_scores: Dict[str, float] = tools.get("afinn_scores", {})
    vader_scores: Dict[str, float] = tools.get("vader_scores", {})
    negation_words: Set[str] = tools.get("negation_words", NEGATION_WORDS_SET)
    booster_words: Dict[str, float] = tools.get("booster_words", BOOSTER_DICT)
    vader_neg_scalar: float = tools.get("vader_neg_scalar", N_SCALAR)
    vader_exclam_boost: float = tools.get("vader_exclam_boost", EXCLAM_BOOST)

    bing_pos_count = 0
    bing_neg_count = 0
    total_afinn_score = 0.0
    total_vader_score = 0.0
    negated_pos_count = 0
    negated_neg_count = 0

    for i, token in enumerate(clean_tokens):
        if token in bing_pos_lex:
            bing_pos_count += 1
        if token in bing_neg_lex:
            bing_neg_count += 1

        score_afinn = afinn_scores.get(token, 0.0)
        score_vader = vader_scores.get(token, 0.0)

        is_negated = False
        for j in range(max(0, i - 3), i):
            if clean_tokens[j] in negation_words:
                is_negated = True
                break

        if score_afinn != 0.0 or score_vader != 0.0:
            scalar = 1.0
            if is_negated:
                scalar *= vader_neg_scalar
                if token in bing_pos_lex:
                    negated_pos_count += 1
                if token in bing_neg_lex:
                    negated_neg_count += 1

            for j in range(max(0, i - 2), i):
                booster_val = booster_words.get(clean_tokens[j], 0.0)
                if booster_val != 0.0:
                    scalar += booster_val

            if i < len(clean_tokens) - 1:
                if clean_tokens[i + 1] == "!" or (
                    i < len(clean_tokens) - 2 and clean_tokens[i + 2] == "!"
                ):
                    if score_vader > 0:
                        score_vader += vader_exclam_boost
                    elif score_vader < 0:
                        score_vader -= vader_exclam_boost
                    if score_afinn > 0:
                        score_afinn += 1.0
                    elif score_afinn < 0:
                        score_afinn -= 1.0

            score_afinn *= scalar
            score_vader *= scalar
        elif is_negated:
            if token in bing_pos_lex:
                negated_pos_count += 1
            if token in bing_neg_lex:
                negated_neg_count += 1

        total_afinn_score += score_afinn
        total_vader_score += score_vader

    bing_liu_net_score = float(bing_pos_count - bing_neg_count)
    lex_score_average = (total_afinn_score + total_vader_score + bing_liu_net_score) / 3.0

    return {
        "__LEX_POS_COUNT__": float(bing_pos_count),
        "__LEX_NEG_COUNT__": float(bing_neg_count),
        "__LEX_SCORE_AFINN__": float(total_afinn_score),
        "__LEX_SCORE_VADER__": float(total_vader_score),
        "__LEX_SCORE_BINGLIU_NET__": bing_liu_net_score,
        "__LEX_SCORE_AVERAGE__": float(lex_score_average),
        "__META_TOKEN_COUNT__": float(len(clean_tokens)),
        "__META_EXCLAM_COUNT__": float(clean_tokens.count("!")),
        "__META_QMARK_COUNT__": float(clean_tokens.count("?")),
        "__META_NEGATED_POS_COUNT__": float(negated_pos_count),
        "__META_NEGATED_NEG_COUNT__": float(negated_neg_count),
    }
