from datasets import load_dataset
from src.utils.log import get_logger
from src.retrievers.base_retriever import Retriever


logger = get_logger()


class ESRetriever(Retriever):
    def __init__(self, data_set: ) -> None:

        pass

    def retrieve(self, query: str, k: int):
        pass
