import os
import os.path

import torch
from datasets import load_dataset
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from src.retrievers.base_retriever import Retriever
from src.utils.log import get_logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Hacky fix for FAISS error on macOS
# See https://stackoverflow.com/a/63374568/4545692


logger = get_logger()


class FAISRetriever(Retriever):
    """A class used to retrieve relevant documents based on some query.
    based on https://huggingface.co/docs/datasets/faiss_es#faiss.
    """

    def __init__(self, dataset_name: str = "GroNLP/ik-nlp-22_slp") -> None:
        """Initialize the retriever

        Args:
            dataset (str, optional): The dataset to train on. Assumes the
            information is stored in a column named 'text'. Defaults to
            "GroNLP/ik-nlp-22_slp".
        """
        torch.set_grad_enabled(False)

        # Context encoding and tokenization
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

        # Question encoding and tokenization
        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

        # Dataset building
        self.dataset_name = dataset_name
        self.dataset = self._init_dataset(dataset_name)

    def _init_dataset(self,
                      dataset_name: str,
                      embedding_path: str = "./models/paragraphs_embedding.faiss"):
        """Loads the dataset and adds FAISS embeddings.

        Args:
            dataset (str): A HuggingFace dataset name.
            fname (str): The name to use to save the embeddings to disk for 
            faster loading after the first run.

        Returns:
            Dataset: A dataset with a new column 'embeddings' containing FAISS
            embeddings.
        """
        # Load dataset
        ds = load_dataset(dataset_name, name="paragraphs")[
            "train"]  # type: ignore
        logger.info(ds)

        if os.path.exists(embedding_path):
            # If we already have FAISS embeddings, load them from disk
            ds.load_faiss_index('embeddings', embedding_path)  # type: ignore
            return ds
        else:
            # If there are no FAISS embeddings, generate them
            def embed(row):
                # Inline helper function to perform embedding
                p = row["text"]
                tok = self.ctx_tokenizer(
                    p, return_tensors="pt", truncation=True)
                enc = self.ctx_encoder(**tok)[0][0].numpy()
                return {"embeddings": enc}

            # Add FAISS embeddings
            ds_with_embeddings = ds.map(embed)  # type: ignore

            ds_with_embeddings.add_faiss_index(column="embeddings")

            # save dataset w/ embeddings
            os.makedirs("./models/", exist_ok=True)
            ds_with_embeddings.save_faiss_index("embeddings", embedding_path)

            return ds_with_embeddings

    def retrieve(self, query: str, k: int = 5):
        """Retrieve the top k matches for a search query.

        Args:
            query (str): A search query
            k (int, optional): The number of documents to retrieve. Defaults to
            5.

        Returns:
            tuple: A tuple of lists of scores and results.
        """

        def embed(q):
            # Inline helper function to perform embedding
            tok = self.q_tokenizer(q, return_tensors="pt", truncation=True)
            return self.q_encoder(**tok)[0][0].numpy()

        question_embedding = embed(query)
        scores, results = self.dataset.get_nearest_examples(
            "embeddings", question_embedding, k=k
        )

        return scores, results
