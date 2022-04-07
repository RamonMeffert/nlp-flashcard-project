import os
import os.path
import torch

from dotenv import load_dotenv
from datasets import DatasetDict
from dataclasses import dataclass
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    LongformerModel,
    LongformerTokenizer
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.retrievers.base_retriever import RetrieveType, Retriever
from src.utils.log import logger
from src.utils.preprocessing import remove_formulas
from src.utils.timing import timeit


load_dotenv()


@dataclass
class FaissRetrieverOptions:
    ctx_encoder: PreTrainedModel
    ctx_tokenizer: PreTrainedTokenizerFast
    q_encoder: PreTrainedModel
    q_tokenizer: PreTrainedTokenizerFast
    embedding_path: str
    lm: str

    @staticmethod
    def dpr(embedding_path: str):
        return FaissRetrieverOptions(
            ctx_encoder=DPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            ),
            ctx_tokenizer=DPRContextEncoderTokenizerFast.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            ),
            q_encoder=DPRQuestionEncoder.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            ),
            q_tokenizer=DPRQuestionEncoderTokenizerFast.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            ),
            embedding_path=embedding_path,
            lm="dpr"
        )

    @staticmethod
    def longformer(embedding_path: str):
        encoder = LongformerModel.from_pretrained(
            "valhalla/longformer-base-4096-finetuned-squadv1"
        )
        tokenizer = LongformerTokenizer.from_pretrained(
            "valhalla/longformer-base-4096-finetuned-squadv1"
        )
        return FaissRetrieverOptions(
            ctx_encoder=encoder,
            ctx_tokenizer=tokenizer,
            q_encoder=encoder,
            q_tokenizer=tokenizer,
            embedding_path=embedding_path,
            lm="longformer"
        )


class FaissRetriever(Retriever):
    """A class used to retrieve relevant documents based on some query.
    based on https://huggingface.co/docs/datasets/faiss_es#faiss.
    """

    def __init__(self, paragraphs: DatasetDict,
                 options: FaissRetrieverOptions) -> None:
        torch.set_grad_enabled(False)

        self.lm = options.lm

        # Context encoding and tokenization
        self.ctx_encoder = options.ctx_encoder
        self.ctx_tokenizer = options.ctx_tokenizer

        # Question encoding and tokenization
        self.q_encoder = options.q_encoder
        self.q_tokenizer = options.q_tokenizer

        self.paragraphs = paragraphs
        self.embedding_path = options.embedding_path

        self.index = self._init_index()

    def _embed_question(self, q):
        match self.lm:
            case "dpr":
                tok = self.q_tokenizer(
                    q, return_tensors="pt", truncation=True, padding=True)
                return self.q_encoder(**tok)[0][0].numpy()
            case "longformer":
                tok = self.q_tokenizer(q, return_tensors="pt")
                return self.q_encoder(**tok).last_hidden_state[0][0].numpy()

    def _embed_context(self, row):
        p = row["text"]

        match self.lm:
            case "dpr":
                tok = self.ctx_tokenizer(
                    p, return_tensors="pt", truncation=True, padding=True)
                enc = self.ctx_encoder(**tok)[0][0].numpy()
                return {"embeddings": enc}
            case "longformer":
                tok = self.ctx_tokenizer(p, return_tensors="pt")
                enc = self.ctx_encoder(**tok).last_hidden_state[0][0].numpy()
                return {"embeddings": enc}

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
            # Add FAISS embeddings
            index = ds.map(self._embed_context)  # type: ignore

            index.add_faiss_index(column="embeddings")

            # save dataset w/ embeddings
            os.makedirs("./src/models/", exist_ok=True)
            index.save_faiss_index(
                "embeddings", self.embedding_path)

            return index

    def retrieve(self, query: str, k: int = 5) -> RetrieveType:
        question_embedding = self._embed_question(query)
        scores, results = self.index.get_nearest_examples(
            "embeddings", question_embedding, k=k
        )

        return scores, results
