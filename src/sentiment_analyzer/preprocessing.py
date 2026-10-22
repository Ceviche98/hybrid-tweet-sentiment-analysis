"""Tweet-aware text normalization and preprocessing pipeline."""

import csv
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import emoji
except ImportError:
    emoji = None  # type: ignore
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

ORIGINAL_NEGATION_WORDS: Set[str] = {
    "no", "nor", "not", "don", "don't", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
}

VADER_NEGATE: List[str] = [
    "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
    "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
    "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neednt",
    "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't",
    "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
    "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
    "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "never",
]

NEGATION_WORDS_SET: Set[str] = ORIGINAL_NEGATION_WORDS.union(set(VADER_NEGATE))

BOOSTER_DICT: Dict[str, float] = {
    "absolutely": 0.293, "amazingly": 0.293, "awfully": 0.293, "completely": 0.293,
    "considerably": 0.293, "decidedly": 0.293, "deeply": 0.293, "effing": 0.293,
    "enormously": 0.293, "entirely": 0.293, "especially": 0.293, "exceptionally": 0.293,
    "extremely": 0.293, "fabulously": 0.293, "flipping": 0.293, "flippin": 0.293,
    "fricking": 0.293, "frickin": 0.293, "frigging": 0.293, "friggin": 0.293, "fully": 0.293,
    "fucking": 0.293, "greatly": 0.293, "hella": 0.293, "highly": 0.293, "hugely": 0.293,
    "incredibly": 0.293, "intensely": 0.293, "majorly": 0.293, "more": 0.293, "most": 0.293,
    "particularly": 0.293, "purely": 0.293, "quite": 0.293, "really": 0.293, "remarkably": 0.293,
    "so": 0.293, "substantially": 0.293, "thoroughly": 0.293, "totally": 0.293,
    "tremendously": 0.293, "uber": 0.293, "unbelievably": 0.293, "unusually": 0.293,
    "utterly": 0.293, "very": 0.293,
    "almost": -0.293, "barely": -0.293, "hardly": -0.293, "just enough": -0.293,
    "kind of": -0.293, "kinda": -0.293, "kindof": -0.293, "kind-of": -0.293,
    "less": -0.293, "little": -0.293, "marginally": -0.293, "occasionally": -0.293,
    "partly": -0.293, "scarcely": -0.293, "slightly": -0.293, "somewhat": -0.293,
    "sort of": -0.293, "sorta": -0.293, "sortof": -0.293, "sort-of": -0.293,
}

N_SCALAR: float = -0.74
EXCLAM_BOOST: float = 0.292

POS_ASCII_EMO_REGEX: str = r"(?::\)|:-\)|=\)|:d|xd|:p|;\)|;\*|;>)"
NEG_ASCII_EMO_REGEX: str = r"(?::\(|:-\(|=\()"
ALL_ASCII_EMO_REGEX: str = (
    r"(:\)|:-\)|=\)|:d|xd|:p|;\)|;\*|;>|:\(|:-\(|=\()|:\||:\[|:\]|:\{|\}:|:o\)|:\'\(|\(::\):|;-\(|;\(|;-\)|;-d|=\(|\)|=\/|=\\|:s|:p|:o\)"
)


def setup_nltk_resources() -> None:
    """Download required NLTK resources safely if not already available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def create_preprocessing_tools(lexicon_tools: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize and return reusable preprocessing and lexicon tools.

    Args:
        lexicon_tools: Optional dictionary of loaded sentiment lexicons.

    Returns:
        Dictionary containing initialized lemmatizer, stemmer, stop words set,
        punctuation set, negation words, booster words, and lexicon structures.
    """
    setup_nltk_resources()
    stop_words_list = stopwords.words("english")
    custom_stop_words_set = set(stop_words_list) - NEGATION_WORDS_SET
    punctuation_to_remove = set(string.punctuation) - {"!", "?"}

    tools: Dict[str, Any] = {
        "lemmatizer": WordNetLemmatizer(),
        "stemmer": PorterStemmer(),
        "stop_words_set": custom_stop_words_set,
        "punctuation_set": punctuation_to_remove,
        "negation_words": NEGATION_WORDS_SET,
        "booster_words": BOOSTER_DICT,
        "vader_neg_scalar": N_SCALAR,
        "vader_exclam_boost": EXCLAM_BOOST,
    }
    if lexicon_tools:
        tools.update(lexicon_tools)
        if "lexicons" not in tools and "bing_liu_positive" in lexicon_tools:
            tools["lexicons"] = {
                "positive": lexicon_tools["bing_liu_positive"],
                "negative": lexicon_tools["bing_liu_negative"],
            }
    return tools


def pre_process_baseline(text: str) -> List[str]:
    """Baseline whitespace tokenizer from Question 1."""
    return text.split()


def normalize_elongated_words(text: str) -> str:
    """Collapse character repetitions of 3+ identical letters down to 2 (e.g. 'soooo' -> 'soo')."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def pre_process_pipeline(
    text: str,
    tools: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Apply modular tweet-aware preprocessing pipeline to raw tweet text.

    Handles case folding, Unicode emoji demojization / separation, ASCII emoticon
    grouping (`__emoticon_pos__` / `__emoticon_neg__`), URL and `@user` mention removal,
    `RT` artifact removal, hashtag normalization, abbreviation expansion (`w/` -> `with`),
    punctuation isolation (`!` and `?`), negation-preserving stopword removal,
    and WordNet lemmatization or Porter stemming.

    Args:
        text: Raw input tweet string.
        tools: Preprocessing tools dictionary from `create_preprocessing_tools`.
        config: Configuration dictionary controlling pipeline stages.

    Returns:
        List of cleaned and normalized string tokens.
    """
    if config is None:
        config = {}

    feature_set_type = config.get("feature_set", "option_1")
    use_lemmatization = config.get("lemmatize", False)
    use_stemming = config.get("stem", False)
    remove_stopwords = config.get("remove_stopwords", False)
    handle_hashtags = config.get("handle_hashtags", "keep_word")
    normalize_elongated = config.get("normalize_elongated", False)

    lemmatizer: WordNetLemmatizer = tools["lemmatizer"]
    stemmer: PorterStemmer = tools["stemmer"]
    stop_words_set: Set[str] = tools["stop_words_set"]
    punctuation_set: Set[str] = tools["punctuation_set"]

    text = text.lower()
    text = re.sub(r"([😀-🙏®-🧿])", r" \1 ", text)

    if feature_set_type == "option_1":
        if emoji is not None:
            text = emoji.demojize(text, delimiters=(" :emoticon_", ": "))
        else:
            for emo_char, desc in tools.get("emoji_descriptions", {}).items():
                if emo_char in text:
                    slug = desc.replace(" ", "_")
                    text = text.replace(emo_char, f" :emoticon_{slug}: ")
        text = re.sub(POS_ASCII_EMO_REGEX, " __emoticon_pos__ ", text)
        text = re.sub(NEG_ASCII_EMO_REGEX, " __emoticon_neg__ ", text)
    elif feature_set_type == "option_2":
        text = re.sub(ALL_ASCII_EMO_REGEX, r" \g<0> ", text)

    # Remove URLs and @user mentions
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    # Remove retweet indicator 'RT'
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)

    # Hashtag handling
    if handle_hashtags == "keep_word":
        text = re.sub(r"#", "", text)
    elif handle_hashtags == "remove":
        text = re.sub(r"#\w+", "", text)

    # Expand abbreviations and isolate sentiment-bearing punctuation
    text = re.sub(r"\bw/\b", "with", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    text = re.sub(r"[\.]{2,}|…", " ", text)

    if normalize_elongated:
        text = normalize_elongated_words(text)

    tokens = word_tokenize(text)

    clean_tokens: List[str] = []
    for token in tokens:
        if token in punctuation_set or token.isnumeric():
            continue
        if remove_stopwords and token in stop_words_set:
            continue
        if use_lemmatization:
            if token == "'s":
                continue
            token = lemmatizer.lemmatize(token)
        elif use_stemming:
            if token == "'s":
                continue
            token = stemmer.stem(token)
        if token == "'s":
            continue
        if token:
            clean_tokens.append(token)

    return clean_tokens


def parse_data_line(data_line: List[str]) -> Tuple[str, str]:
    """Extract (label, text) from a tab-separated dataset row."""
    return (data_line[1], data_line[2])


def load_and_split_raw_data(
    path: Union[str, Path],
    percentage: float = 0.8,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load tab-separated sentiment dataset and split into train and test sets.

    Args:
        path: Path to `sentiment-dataset.tsv`.
        percentage: Fraction of dataset allocated to training (default 0.8).

    Returns:
        Tuple of `(raw_train_data, raw_test_data)` where each item is `(text, label)`.
    """
    raw_data: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) >= 3:
                label, text = parse_data_line(line)
                if label and text:
                    raw_data.append((text, label))

    num_samples = len(raw_data)
    num_training_samples = int(percentage * num_samples)
    raw_train_data = raw_data[:num_training_samples]
    raw_test_data = raw_data[num_training_samples:]
    return raw_train_data, raw_test_data
