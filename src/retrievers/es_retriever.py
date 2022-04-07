import imp
import os

from datasets import DatasetDict
from elasticsearch import Elasticsearch
from elastic_transport import ConnectionError
from dotenv import load_dotenv

from src.retrievers.base_retriever import RetrieveType, Retriever
from src.utils.log import logger
from src.utils.timing import timeit


load_dotenv()


class ESRetriever(Retriever):
    def __init__(self, paragraphs: DatasetDict) -> None:
        self.paragraphs = paragraphs["train"]

        es_host = os.getenv("ELASTIC_HOST", "localhost")
        es_password = os.getenv("ELASTIC_PASSWORD")
        es_username = os.getenv("ELASTIC_USERNAME")

        self.client = Elasticsearch(
            hosts=[es_host],
            http_auth=(es_username, es_password),
            ca_certs="./http_ca.crt")

        try:
            self.client.info()
        except ConnectionError:
            logger.error("Could not connect to ElasticSearch. " + 
                         "Make sure it is running. Exiting now...")
            exit()

        if self.client.indices.exists(index="paragraphs"):
            self.paragraphs.load_elasticsearch_index(
                "paragraphs", es_index_name="paragraphs",
                es_client=self.client)
        else:
            logger.info(f"Creating index 'paragraphs' on {es_host}")
            self.paragraphs.add_elasticsearch_index(column="text",
                                                    index_name="paragraphs",
                                                    es_index_name="paragraphs",
                                                    es_client=self.client)

    def retrieve(self, query: str, k: int = 5) -> RetrieveType:
        return self.paragraphs.get_nearest_examples("paragraphs", query, k)
