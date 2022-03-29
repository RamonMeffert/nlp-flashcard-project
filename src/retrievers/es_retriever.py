from datasets import DatasetDict, load_dataset
from src.utils.log import get_logger
from src.retrievers.base_retriever import Retriever
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

logger = get_logger()


class ESRetriever(Retriever):
    def __init__(self, dataset: DatasetDict) -> None:
        self.dataset = dataset["train"]

        es_host = os.getenv("ELASTIC_HOST", "localhost")
        es_password = os.getenv("ELASTIC_PASSWORD")
        es_username = os.getenv("ELASTIC_USERNAME")

        self.client = Elasticsearch(
            hosts=[es_host],
            http_auth=(es_username, es_password),
            ca_certs="./http_ca.crt")

        if self.client.indices.exists(index="paragraphs"):
            self.dataset.load_elasticsearch_index(
                "paragraphs", es_index_name="paragraphs",
                es_client=self.client)
        else:
            logger.info(f"Creating index 'paragraphs' on {es_host}")
            self.dataset.add_elasticsearch_index(column="text",
                                                 index_name="paragraphs",
                                                 es_index_name="paragraphs",
                                                 es_client=self.client)

    def retrieve(self, query: str, k: int = 5):
        return self.dataset.get_nearest_examples("paragraphs", query, k)
