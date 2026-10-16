import math
import re
import logging
from typing import List, Dict, Any, Set
from app.models.retrieval import RetrievedChunk
from app.models.rerank import RankedChunk
from app.core.config import settings

logger = logging.getLogger(__name__)

class AlgorithmicRRFReranker:
    """
    Zero-Memory Algorithmic Reciprocal Rank Fusion (RRF) & Lexical Score Fusion Engine.
    
    Eliminates all local heavy neural cross-encoders (FlashRank, PyTorch, ONNX) to prevent
    Render Free-Tier 512 MB Out-Of-Memory (OOM) crashes.
    
    Executes in < 1 ms with 0 MB model overhead.
    """

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple, fast regex token extractor for lexical overlap."""
        return set(re.findall(r'\b[a-z0-9_]{3,}\b', text.lower()))

    @classmethod
    def rerank(
        cls,
        query: str,
        candidate_chunks: List[RetrievedChunk],
        top_n: int = 6,
        score_threshold: float = 0.05,
        cormack_k: int = 60
    ) -> List[RankedChunk]:
        """
        Fuses native Qdrant RRF retrieval ranks with exact keyword density and heading boosts:
        Score = RRF_Rank_Score + Lexical_Jaccard_Bonus + Heading_Match_Bonus
        """
        if not candidate_chunks:
            return []

        query_tokens = cls._tokenize(query)
        scored_candidates: List[tuple[float, RetrievedChunk]] = []

        # Candidate chunks arrive ordered by Qdrant hybrid prefetch RRF
        for rank_idx, chunk in enumerate(candidate_chunks):
            # 1. Base Cormack RRF score: 1 / (k + rank)
            # Higher rank (0, 1, 2...) gets highest score
            rrf_score = 1.0 / (cormack_k + (rank_idx + 1))

            # 2. Pure Python Lexical Overlap Bonus
            chunk_tokens = cls._tokenize(chunk.text)
            if query_tokens and chunk_tokens:
                intersection = query_tokens.intersection(chunk_tokens)
                jaccard = len(intersection) / len(query_tokens.union(chunk_tokens))
                coverage = len(intersection) / len(query_tokens)
            else:
                jaccard = 0.0
                coverage = 0.0

            # 3. Structural Heading Match Bonus
            heading_boost = 0.0
            if chunk.section_header:
                heading_tokens = cls._tokenize(chunk.section_header)
                if query_tokens.intersection(heading_tokens):
                    heading_boost = 0.15

            # 4. Fused Composite Score
            # Scale RRF (typically ~0.016) to comparable range, add lexical density
            fused_score = (rrf_score * 30.0) + (coverage * 0.40) + (jaccard * 0.30) + heading_boost
            
            # Normalize reasonably into [0.20, 0.99]
            normalized_score = min(0.99, max(0.20, fused_score))
            scored_candidates.append((normalized_score, chunk))

        # Sort descending by fused score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        ranked_list: List[RankedChunk] = []
        for score, chunk in scored_candidates:
            if score >= score_threshold:
                ranked_list.append(
                    RankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        section_header=chunk.section_header,
                        has_table=chunk.has_table,
                        has_image=chunk.has_image,
                        relevance_score=round(score, 4),
                        metadata=chunk.metadata
                    )
                )

        final_ranked = ranked_list[:top_n]
        logger.info(
            f"Algorithmic RRF reranked {len(candidate_chunks)} candidates -> Kept {len(final_ranked)} chunks "
            f"(top_n={top_n}, 0 MB RAM allocated)"
        )
        return final_ranked

algorithmic_reranker = AlgorithmicRRFReranker()
