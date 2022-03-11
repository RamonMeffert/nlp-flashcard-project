from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, \
                         DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import load_dataset
import torch


class Retriever():
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
            "facebook/dpr-ctx_encoder-single-nq-base")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base")

        # Question encoding and tokenization
        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base")

        # Dataset building
        self.dataset = self.__init_dataset(dataset)

    def __init_dataset(self, dataset: str):
        """Loads the dataset and adds FAISS embeddings.

        Args:
            dataset (str): A HuggingFace dataset name.

        Returns:
            Dataset: A dataset with a new column 'embeddings' containing FAISS
            embeddings.
        """
        # TODO: save ds w/ embeddings to disk and retrieve it if it already exists

        # Load dataset
        ds = load_dataset(dataset, name='paragraphs')['train']

        def embed(row):
            # Inline helper function to perform embedding
            p = row['text']
            tok = self.ctx_tokenizer(p, return_tensors='pt', truncation=True)
            enc = self.ctx_encoder(**tok)[0][0].numpy()
            return {'embeddings': enc}

        # Add FAISS embeddings
        ds_with_embeddings = ds.map(embed)

        # Todo: this throws a weird error.
        ds_with_embeddings.add_faiss_index(column='embeddings')
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
            tok = self.q_tokenizer(q, return_tensors='pt', truncation=True)
            return self.q_encoder(**tok)[0][0].numpy()

        question_embedding = embed(query)
        scores, results = self.dataset.get_nearest_examples(
            'embeddings', question_embedding, k=k)
        return scores, results
