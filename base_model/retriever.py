from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
from datasets import load_dataset
import torch
import os.path

# Hacky fix for FAISS error on macOS
# See https://stackoverflow.com/a/63374568/4545692
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Retriever:
    """A class used to retrieve relevant documents based on some query.
    based on https://huggingface.co/docs/datasets/faiss_es#faiss.
    """

    def __init__(self, dataset: str = "GroNLP/ik-nlp-22_slp") -> None:
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
        self.dataset = self.__init_dataset(dataset)

    def __init_dataset(self,
                       dataset: str,
                       fname: str = "./models/paragraphs_embedding.faiss"):
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
        ds = load_dataset(dataset, name="paragraphs")["train"]

        if os.path.exists(fname):
            # If we already have FAISS embeddings, load them from disk
            ds.load_faiss_index('embeddings', fname)
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
            ds_with_embeddings = ds.map(embed)

            ds_with_embeddings.add_faiss_index(column="embeddings")

            # save dataset w/ embeddings
            os.makedirs("./models/", exist_ok=True)
            ds_with_embeddings.save_faiss_index("embeddings", fname)

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
