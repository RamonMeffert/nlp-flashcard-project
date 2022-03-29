import argparse
import torch
import transformers

from typing import List
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv

from src.readers.dpr_reader import DprReader
from src.retrievers.base_retriever import Retriever
from src.retrievers.es_retriever import ESRetriever
from src.retrievers.faiss_retriever import FaissRetriever
from src.utils.preprocessing import result_to_reader_input
from src.utils.log import get_logger


def get_retriever(r: str, ds: DatasetDict) -> Retriever:
    retriever = ESRetriever if r == "es" else FaissRetriever
    return retriever(ds)


def print_name(contexts: dict, section: str, id: int):
    name = contexts[section][id]
    if name != 'nan':
        print(f"      {section}: {name}")


def print_answers(answers: List[tuple], scores: List[float], contexts: dict):
    # calculate answer scores
    sm = torch.nn.Softmax(dim=0)
    d_scores = sm(torch.Tensor(
        [pred.relevance_score for pred in answers]))
    s_scores = sm(torch.Tensor(
        [pred.span_score for pred in answers]))

    for pos, answer in enumerate(answers):
        print(f"{pos + 1:>4}. {answer.text}")
        print(f"      {'-' * len(answer.text)}")
        print_name(contexts, 'chapter', answer.doc_id)
        print_name(contexts, 'section', answer.doc_id)
        print_name(contexts, 'subsection', answer.doc_id)
        print(f"      retrieval score: {scores[answer.doc_id]:6.02f}%")
        print(f"      document score:  {d_scores[pos] * 100:6.02f}%")
        print(f"      span score:      {s_scores[pos] * 100:6.02f}%")
        print()


def main(args: argparse.Namespace):
    # Initialize dataset
    dataset = load_dataset("GroNLP/ik-nlp-22_slp")

    # Retrieve
    retriever = get_retriever(args.retriever, dataset)
    scores, contexts = retriever.retrieve(args.query)

    # Read
    reader = DprReader()
    reader_input = result_to_reader_input(contexts)
    answers = reader.read(args.query, reader_input, num_answers=args.top)

    # Print output
    print_answers(answers, scores, contexts)


if __name__ == "__main__":
    # Setup environment
    load_dotenv()
    logger = get_logger()
    transformers.logging.set_verbosity_error()

    # Set up CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    parser.add_argument("query", type=str,
                        help="The question to feed to the QA system")
    parser.add_argument("--top", "-t", type=int, default=1,
                        help="The number of answers to retrieve")
    parser.add_argument("--retriever", "-r", type=str.lower,
                        choices=["faiss", "es"], default="faiss",
                        help="The retrieval method to use")

    args = parser.parse_args()
    main(args)
