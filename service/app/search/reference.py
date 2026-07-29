from __future__ import annotations

import numpy as np


def common_exact(embeddings: np.ndarray, query: np.ndarray, ids: list[str], top_k: int, mask: np.ndarray | None = None) -> list[tuple[str,float]]:
    matrix=np.asarray(embeddings,dtype=np.float32); q=np.asarray(query,dtype=np.float32)
    if mask is not None:
        selected=np.flatnonzero(mask); matrix=matrix[selected]; selected_ids=[ids[i] for i in selected]
    else: selected_ids=ids
    if not len(matrix): return []
    scores=np.sum(matrix*q[None,:],axis=1,dtype=np.float32); order=np.argsort(-scores,kind="stable")[:top_k]
    return [(selected_ids[i],float(scores[i])) for i in order]


def quality_fields(n_ground_truth: int, returned_count: int) -> dict:
    if n_ground_truth == 0:
        return {"r_at_1":None,"ndcg":None,"quality_vs_groundtruth":None,"returned_count":returned_count}
    return {"r_at_1":0.0,"ndcg":0.0,"quality_vs_groundtruth":0.0,"returned_count":returned_count}
