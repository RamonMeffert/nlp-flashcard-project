import os
import os.path

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from src.retrievers.base_retriever import RetrieveType, Retriever
from src.utils.log import get_logger
from src.utils.preprocessing import remove_formulas
from src.utils.timing import timeit

# Hacky fix for FAISS error on macOS
# See https://stackoverflow.com/a/63374568/4545692
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


logger = get_logger()


class FaissRetriever(Retriever):
    """A class used to retrieve relevant documents based on some query.
    based on https://huggingface.co/docs/datasets/faiss_es#faiss.
    """

    def __init__(self, paragraphs: DatasetDict, embedding_path: str = "./src/models/paragraphs_embedding.faiss") -> None:
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

        self.paragraphs = paragraphs
        self.embedding_path = embedding_path

        self.index = self._init_index()

    def _init_index(
            self,
            force_new_embedding: bool = False):

        ds = self.paragraphs["train"]
        ds = ds.map(remove_formulas)

        if not force_new_embedding and os.path.exists(self.embedding_path):
            ds.load_faiss_index(
                'embeddings', self.embedding_path)  # type: ignore
            return ds
        else:
            def embed(row):
                # Inline helper function to perform embedding
                p = row["text"]
                tok = self.ctx_tokenizer(
                    p, return_tensors="pt", truncation=True)
                enc = self.ctx_encoder(**tok)[0][0].numpy()
                return {"embeddings": enc}

            # Add FAISS embeddings
            index = ds.map(embed)  # type: ignore

            index.add_faiss_index(column="embeddings")

            # save dataset w/ embeddings
            os.makedirs("./src/models/", exist_ok=True)
            index.save_faiss_index(
                "embeddings", self.embedding_path)

            return index

    @timeit("faissretriever.retrieve")
    def retrieve(self, query: str, k: int = 5) -> RetrieveType:
        def embed(q):
            # Inline helper function to perform embedding
            tok = self.q_tokenizer(q, return_tensors="pt", truncation=True)
            return self.q_encoder(**tok)[0][0].numpy()

        question_embedding = embed(query)
        scores, results = self.index.get_nearest_examples(
            "embeddings", question_embedding, k=k
        )

        return scores, results
