import argparse
import torch
import transformers

from typing import Dict, List, Literal, Tuple, cast
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv

from src.readers.base_reader import Reader
from src.readers.longformer_reader import LongformerReader
from src.readers.dpr_reader import DprReader
from src.retrievers.base_retriever import Retriever
from src.retrievers.es_retriever import ESRetriever
from src.retrievers.faiss_retriever import (
    FaissRetriever,
    FaissRetrieverOptions
)
from src.utils.preprocessing import context_to_reader_input
from src.utils.log import get_logger


def get_retriever(paragraphs: DatasetDict,
                  r: Literal["es", "faiss"],
                  lm: Literal["dpr", "longformer"]) -> Retriever:
    match (r, lm):
        case "es", _:
            return ESRetriever()
        case "faiss", "dpr":
            options = FaissRetrieverOptions.dpr("./src/models/dpr.faiss")
            return FaissRetriever(paragraphs, options)
        case "faiss", "longformer":
            options = FaissRetrieverOptions.longformer(
                "./src/models/longformer.faiss")
            return FaissRetriever(paragraphs, options)
        case _:
            raise ValueError("Retriever options not recognized")


def get_reader(lm: Literal["dpr", "longformer"]) -> Reader:
    match lm:
        case "dpr":
            return DprReader()
        case "longformer":
            return LongformerReader()
        case _:
            raise ValueError("Language model not recognized")


def print_name(contexts: dict, section: str, id: int):
    name = contexts[section][id]
    if name != 'nan':
        print(f"      {section}: {name}")


def get_retrieval_span_scores(answers: List[tuple]):
    # calculate answer scores
    sm = torch.nn.Softmax(dim=0)
    d_scores = sm(torch.Tensor(
        [pred.relevance_score for pred in answers]))
    s_scores = sm(torch.Tensor(
        [pred.span_score for pred in answers]))

    return d_scores, s_scores


def print_answers(answers: List[tuple], scores: List[float], contexts: dict):
    d_scores, s_scores = get_retrieval_span_scores(answers)

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


def probe(query: str,
          retriever: Retriever,
          reader: Reader,
          num_answers: int = 5) \
          -> Tuple[List[tuple], List[float], Dict[str, List[str]]]:
    scores, contexts = retriever.retrieve(query)
    reader_input = context_to_reader_input(contexts)
    answers = reader.read(query, reader_input, num_answers)

    return answers, scores, contexts


def default_probe(query: str):
    # default probe is a probe that prints 5 answers with faiss
    paragraphs = cast(DatasetDict, load_dataset(
        "GroNLP/ik-nlp-22_slp", "paragraphs"))
    retriever = get_retriever(paragraphs, "faiss", "dpr")
    reader = DprReader()

    return probe(query, retriever, reader)


def main(args: argparse.Namespace):
    # Initialize dataset
    paragraphs = cast(DatasetDict, load_dataset(
        "GroNLP/ik-nlp-22_slp", "paragraphs"))

    # Retrieve
    retriever = get_retriever(paragraphs, args.retriever, args.lm)
    reader = get_reader(args.lm)
    answers, scores, contexts = probe(
        args.query, retriever, reader, args.top)

    # Print output
    print("Question: " + args.query)
    print("Answer(s):")
    if args.lm == "dpr":
        print_answers(answers, scores, contexts)
    else:
        answers = filter(lambda a: len(a[0].strip()) > 0, answers)
        for pos, answer in enumerate(answers, start=1):
            print(f"    - {answer[0].strip()}")


if __name__ == "__main__":
    # Setup environment
    load_dotenv()
    logger = get_logger()
    transformers.logging.set_verbosity_error()

    # Set up CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    parser.add_argument(
        "query", type=str, help="The question to feed to the QA system")
    parser.add_argument(
        "--top", "-t", type=int, default=1,
        help="The number of answers to retrieve")
    parser.add_argument(
        "--retriever", "-r", type=str.lower, choices=["faiss", "es"],
        default="faiss", help="The retrieval method to use")
    parser.add_argument(
        "--lm", "-l", type=str.lower,
        choices=["dpr", "longformer"], default="dpr",
        help="The language model to use for the FAISS retriever")

    args = parser.parse_args()
    main(args)
