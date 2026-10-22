"""Dense semantic representations using SentenceTransformer fused with LexV2 features."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sentiment_analyzer.lexicons import META_FEATURE_NAMES_V2, extract_lexv2_features
from sentiment_analyzer.preprocessing import pre_process_pipeline

SUPPORTED_ST_MODELS: Tuple[str, ...] = (
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
)


class HybridEmbeddingExtractor:
    """Combines pretrained SentenceTransformer dense embeddings with LexV2 sentiment features.

    In our benchmark ablation study, concatenating 768-dimensional `all-mpnet-base-v2`
    embeddings with the 11-dimensional `LexV2` feature vector (779 total features)
    trained with a DART-boosted LightGBM classifier achieved the cohort-leading
    91.04% 10-fold CV F1-score and 91.00% test F1-score.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        tools: Optional[Dict[str, Any]] = None,
        preproc_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.tools = tools or {}
        self.preproc_config = preproc_config or {
            "feature_set": "option_2",
            "lemmatize": True,
            "remove_stopwords": True,
            "handle_hashtags": "keep_word",
        }
        self.device = device
        self._st_model: Any = None
        self.meta_vectorizer = DictVectorizer(sparse=False)
        self._is_fitted = False

    def _load_st_model(self) -> Any:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name, device=self.device)
        return self._st_model

    def extract_lexv2_matrix(
        self,
        raw_texts: List[str],
        fit_vectorizer: bool = False,
    ) -> np.ndarray:
        """Compute the 11-dimensional LexV2 feature matrix for a list of raw tweets."""
        meta_dicts: List[Dict[str, float]] = []
        for text in raw_texts:
            clean_tokens = pre_process_pipeline(text, self.tools, self.preproc_config)
            lex_feats = extract_lexv2_features(clean_tokens, self.tools)
            # Ensure all 11 canonical keys are present
            ordered_feats = {k: lex_feats.get(k, 0.0) for k in META_FEATURE_NAMES_V2}
            meta_dicts.append(ordered_feats)

        if fit_vectorizer or not self._is_fitted:
            self.meta_vectorizer.fit([{k: 1.0 for k in META_FEATURE_NAMES_V2}])
            self._is_fitted = True

        return np.asarray(self.meta_vectorizer.transform(meta_dicts), dtype=np.float32)

    def encode_sentences(
        self,
        raw_texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode raw tweets into dense semantic vectors using SentenceTransformer."""
        st_model = self._load_st_model()
        embeddings = st_model.encode(
            raw_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit_transform(
        self,
        raw_texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Compute concatenated `[SentenceTransformer_Embeddings || LexV2_Features]` matrix."""
        embed_matrix = self.encode_sentences(
            raw_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        meta_matrix = self.extract_lexv2_matrix(raw_texts, fit_vectorizer=True)
        return np.hstack([embed_matrix, meta_matrix])

    def transform(
        self,
        raw_texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Transform unseen raw tweets into the hybrid `[ST || LexV2]` representation."""
        embed_matrix = self.encode_sentences(
            raw_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        meta_matrix = self.extract_lexv2_matrix(raw_texts, fit_vectorizer=False)
        return np.hstack([embed_matrix, meta_matrix])
