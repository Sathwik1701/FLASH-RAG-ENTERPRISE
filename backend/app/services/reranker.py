import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Ultra-low latency FlashRank Reranker Service.
    Uses quantized TinyBERT (3.2 MB) running in ~40 milliseconds on CPU.
    Completely eliminates multi-gigabyte HuggingFace downloads and token authentication warnings.
    """

    def __init__(self):
        self._ranker = None

    @property
    def ranker(self):
        if self._ranker is None:
            from flashrank import Ranker
            logger.info("Initializing ultra-fast FlashRank Ranker (ms-marco-TinyBERT-L-2-v2)...")
            self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            logger.info("FlashRank Ranker loaded successfully.")
        return self._ranker

    def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """
        Reranks passages against query in ~40ms.
        Returns normalized relative scores in [0.0, 1.0].
        """
        if not passages:
            return []

        try:
            from flashrank import RerankRequest
            formatted_passages = [{"id": idx, "text": text} for idx, text in enumerate(passages)]
            request = RerankRequest(query=query, passages=formatted_passages)
            results = self.ranker.rerank(request)

            raw_scores = [float(res["score"]) for res in results]
            max_s = max(raw_scores) if raw_scores else 1.0
            min_s = min(raw_scores) if raw_scores else 0.0
            range_s = max_s - min_s if max_s > min_s else 1.0

            # Map back to original passage indices and normalize relatively to [0.0, 1.0]
            score_map = {}
            for res in results:
                raw_val = float(res["score"])
                # Normalize between 0.35 and 0.95 relative to batch distribution
                norm_val = 0.35 + 0.60 * ((raw_val - min_s) / range_s)
                score_map[res["id"]] = norm_val

            return [score_map.get(i, 0.5) for i in range(len(passages))]

        except Exception as e:
            logger.warning(f"FlashRank reranking error: {e}. Falling back to default scoring.", exc_info=True)
            return [0.8 for _ in passages]

reranker_service = RerankerService()
