from typing import Dict, List, Tuple

import numpy as np

RetrieveTypeResult = Dict[str, List[str]]
RetrieveTypeScores = np.ndarray
RetrieveType = Tuple[RetrieveTypeScores, RetrieveTypeResult]


class Retriever():
    def retrieve(self, query: str, k: int) -> RetrieveType:
        raise NotImplementedError()
