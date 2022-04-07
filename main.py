from dotenv import load_dotenv
# needs to happen as very first thing, otherwise HF ignores env vars
load_dotenv()

import os
import pandas as pd

from dataclasses import dataclass
from typing import Dict, cast
from datasets import DatasetDict, load_dataset

from src.readers.base_reader import Reader
from src.evaluation import evaluate
from src.readers.dpr_reader import DprReader
from src.readers.longformer_reader import LongformerReader
from src.retrievers.base_retriever import Retriever
from src.retrievers.es_retriever import ESRetriever
from src.retrievers.faiss_retriever import (
    FaissRetriever,
    FaissRetrieverOptions
)
from src.utils.log import logger
from src.utils.preprocessing import context_to_reader_input
from src.utils.timing import get_times, timeit


@dataclass
class Experiment:
    retriever: Retriever
    reader: Reader


if __name__ == '__main__':
    dataset_name = "GroNLP/ik-nlp-22_slp"
    paragraphs = cast(DatasetDict, load_dataset(
        "GroNLP/ik-nlp-22_slp", "paragraphs"))
    questions = cast(DatasetDict, load_dataset(dataset_name, "questions"))

    # Only doing a few questions for speed
    subset_idx = len(questions["test"])
    questions_test = questions["test"][:subset_idx]

    experiments: Dict[str, Experiment] = {
        "faiss_dpr": Experiment(
            retriever=FaissRetriever(
                paragraphs,
                FaissRetrieverOptions.dpr("./src/models/dpr.faiss")),
            reader=DprReader()
        ),
        "faiss_longformer": Experiment(
            retriever=FaissRetriever(
                paragraphs,
                FaissRetrieverOptions.longformer("./src/models/longformer.faiss")),
            reader=LongformerReader()
        ),
        "es_dpr": Experiment(
            retriever=ESRetriever(paragraphs),
            reader=DprReader()
        ),
        "es_longformer": Experiment(
            retriever=ESRetriever(paragraphs),
            reader=LongformerReader()
        ),
    }

    for experiment_name, experiment in experiments.items():
        logger.info(f"Running experiment {experiment_name}...")
        for idx in range(subset_idx):
            question = questions_test["question"][idx]
            answer = questions_test["answer"][idx]

            retrieve_timer = timeit(f"{experiment_name}.retrieve")
            t_retrieve = retrieve_timer(experiment.retriever.retrieve)

            read_timer = timeit(f"{experiment_name}.read")
            t_read = read_timer(experiment.reader.read)

            print(f"\x1b[1K\r[{idx+1:03}] - \"{question}\"", end='')

            scores, context = t_retrieve(question, 5)
            reader_input = context_to_reader_input(context)

            # workaround so we can use the decorator with a dynamic name for
            # time recording
            answers = t_read(question, reader_input, 5)

            # Calculate softmaxed scores for readable output
            # sm = torch.nn.Softmax(dim=0)
            # document_scores = sm(torch.Tensor(
            #     [pred.relevance_score for pred in answers]))
            # span_scores = sm(torch.Tensor(
            #     [pred.span_score for pred in answers]))

            # print_answers(answers, scores, context)

            # TODO evaluation and storing of results
        print()

    times = get_times()

    df = pd.DataFrame(times)
    os.makedirs("./results/", exist_ok=True)
    df.to_csv("./results/timings.csv")


    # TODO evaluation and storing of results

    # # Initialize retriever
    # retriever = FaissRetriever(paragraphs)
    # # retriever = ESRetriever(paragraphs)

    # # Retrieve example
    # # random.seed(111)
    # random_index = random.randint(0, len(questions_test["question"])-1)
    # example_q = questions_test["question"][random_index]
    # example_a = questions_test["answer"][random_index]

    # scores, result = retriever.retrieve(example_q)
    # reader_input = context_to_reader_input(result)

    # # TODO: use new code from query.py to clean this up
    # # Initialize reader
    # answers = reader.read(example_q, reader_input)

    # # Calculate softmaxed scores for readable output
    # sm = torch.nn.Softmax(dim=0)
    # document_scores = sm(torch.Tensor(
    #     [pred.relevance_score for pred in answers]))
    # span_scores = sm(torch.Tensor(
    #     [pred.span_score for pred in answers]))

    # print(example_q)
    # for answer_i, answer in enumerate(answers):
    #     print(f"[{answer_i + 1}]: {answer.text}")
    #     print(f"\tDocument {answer.doc_id}", end='')
    #     print(f"\t(score {document_scores[answer_i] * 100:.02f})")
    #     print(f"\tSpan {answer.start_index}-{answer.end_index}", end='')
    #     print(f"\t(score {span_scores[answer_i] * 100:.02f})")
    #     print()  # Newline

    # # print(f"Example q: {example_q} answer: {result['text'][0]}")

    # # for i, score in enumerate(scores):
    # #     print(f"Result {i+1} (score: {score:.02f}):")
    # #     print(result['text'][i])

    # # Determine best answer we want to evaluate
    # highest, highest_index = 0, 0
    # for i, value in enumerate(span_scores):
    #     if value + document_scores[i] > highest:
    #         highest = value + document_scores[i]
    #         highest_index = i

    # # Retrieve exact match and F1-score
    # exact_match, f1_score = evaluate(
    #     example_a, answers[highest_index].text)
    # print(f"Gold answer: {example_a}\n"
    #       f"Predicted answer: {answers[highest_index].text}\n"
    #       f"Exact match: {exact_match:.02f}\n"
    #       f"F1-score: {f1_score:.02f}")

    # Calculate overall performance
    # total_f1 = 0
    # total_exact = 0
    # total_len = len(questions_test["question"])
    # start_time = time.time()
    # for i, question in enumerate(questions_test["question"]):
    #     print(question)
    #     answer = questions_test["answer"][i]
    #     print(answer)
    #
    #     scores, result = retriever.retrieve(question)
    #     reader_input = result_to_reader_input(result)
    #     answers = reader.read(question, reader_input)
    #
    #     document_scores = sm(torch.Tensor(
    #         [pred.relevance_score for pred in answers]))
    #     span_scores = sm(torch.Tensor(
    #         [pred.span_score for pred in answers]))
    #
    #     highest, highest_index = 0, 0
    #     for j, value in enumerate(span_scores):
    #         if value + document_scores[j] > highest:
    #             highest = value + document_scores[j]
    #             highest_index = j
    #     print(answers[highest_index])
    #     exact_match, f1_score = evaluate(answer, answers[highest_index].text)
    #     total_f1 += f1_score
    #     total_exact += exact_match
    # print(f"Total time:", round(time.time() - start_time, 2), "seconds.")
    # print(total_f1)
    # print(total_exact)
    # print(total_f1/total_len)
