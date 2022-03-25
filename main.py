import os
import random
from typing import cast

import torch
import transformers
from datasets import DatasetDict, load_dataset

from src.evaluation import evaluate
from src.readers.dpr_reader import DprReader
from src.retrievers.es_retriever import ESRetriever
from src.retrievers.faiss_retriever import FaissRetriever
from src.utils.log import get_logger
from src.utils.preprocessing import result_to_reader_input

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

logger = get_logger()
transformers.logging.set_verbosity_error()

if __name__ == '__main__':
    dataset_name = "GroNLP/ik-nlp-22_slp"
    paragraphs = load_dataset(dataset_name, "paragraphs")
    questions = cast(DatasetDict, load_dataset(dataset_name, "questions"))

    questions_test = questions["test"]

    # logger.info(questions)

    dataset_paragraphs = cast(DatasetDict, load_dataset(
        "GroNLP/ik-nlp-22_slp", "paragraphs"))

    # Initialize retriever
    # retriever = FaissRetriever(dataset_paragraphs)
    retriever = ESRetriever(dataset_paragraphs)

    # Retrieve example
    # random.seed(111)
    random_index = random.randint(0, len(questions_test["question"])-1)
    example_q = questions_test["question"][random_index]
    example_a = questions_test["answer"][random_index]

    scores, result = retriever.retrieve(example_q)
    reader_input = result_to_reader_input(result)

    # Initialize reader
    reader = DprReader()
    answers = reader.read(example_q, reader_input)

    # Calculate softmaxed scores for readable output
    sm = torch.nn.Softmax(dim=0)
    document_scores = sm(torch.Tensor(
        [pred.relevance_score for pred in answers]))
    span_scores = sm(torch.Tensor(
        [pred.span_score for pred in answers]))

    print(example_q)
    for answer_i, answer in enumerate(answers):
        print(f"[{answer_i + 1}]: {answer.text}")
        print(f"\tDocument {answer.doc_id}", end='')
        print(f"\t(score {document_scores[answer_i] * 100:.02f})")
        print(f"\tSpan {answer.start_index}-{answer.end_index}", end='')
        print(f"\t(score {span_scores[answer_i] * 100:.02f})")
        print()  # Newline

    # print(f"Example q: {example_q} answer: {result['text'][0]}")

    # for i, score in enumerate(scores):
    #     print(f"Result {i+1} (score: {score:.02f}):")
    #     print(result['text'][i])

    # Determine best answer we want to evaluate
    highest, highest_index = 0, 0
    for i, value in enumerate(span_scores):
        if value + document_scores[i] > highest:
            highest = value + document_scores[i]
            highest_index = i

    # Retrieve exact match and F1-score
    exact_match, f1_score = evaluate(
        example_a, answers[highest_index].text)
    print(f"Gold answer: {example_a}\n"
          f"Predicted answer: {answers[highest_index].text}\n"
          f"Exact match: {exact_match:.02f}\n"
          f"F1-score: {f1_score:.02f}")
