from collections import namedtuple
from pprint import pprint
from dotenv import load_dotenv
# needs to happen as very first thing, otherwise HF ignores env vars
load_dotenv()

import os
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, cast, List
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


ExperimentResult = namedtuple('ExperimentResult', ['correct', 'given'])


@dataclass
class Experiment:
    retriever: Retriever
    reader: Reader
    lm: str
    results: List[ExperimentResult] = field(default_factory=list)


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
            reader=DprReader(),
            lm="dpr"
        ),
        "faiss_longformer": Experiment(
            retriever=FaissRetriever(
                paragraphs,
                FaissRetrieverOptions.longformer("./src/models/longformer.faiss")),
            reader=LongformerReader(),
            lm="longformer"
        ),
        "es_dpr": Experiment(
            retriever=ESRetriever(paragraphs),
            reader=DprReader(),
            lm="dpr"
        ),
        "es_longformer": Experiment(
            retriever=ESRetriever(paragraphs),
            reader=LongformerReader(),
            lm="longformer"
        ),
    }

    for experiment_name, experiment in experiments.items():
        logger.info(f"Running experiment {experiment_name}...")
        for idx in range(subset_idx):
            question = questions_test["question"][idx]
            answer = questions_test["answer"][idx]

            # workaround so we can use the decorator with a dynamic name for
            # time recording
            retrieve_timer = timeit(f"{experiment_name}.retrieve")
            t_retrieve = retrieve_timer(experiment.retriever.retrieve)

            read_timer = timeit(f"{experiment_name}.read")
            t_read = read_timer(experiment.reader.read)

            print(f"\x1b[1K\r[{idx+1:03}] - \"{question}\"", end='')

            scores, context = t_retrieve(question, 5)
            reader_input = context_to_reader_input(context)

            # Requesting 1 answers results in us getting the best answer
            given_answer = t_read(question, reader_input, 1)[0]

            # Save the results so we can evaluate laters
            if experiment.lm == "longformer":
                experiment.results.append(
                    ExperimentResult(answer, given_answer[0]))
            else:
                experiment.results.append(
                    ExperimentResult(answer, given_answer.text))

        print()

    if os.getenv("ENABLE_TIMING", "false").lower() == "true":
        # Save times
        times = get_times()
        df = pd.DataFrame(times)
        os.makedirs("./results/", exist_ok=True)
        df.to_csv("./results/timings.csv")

    f1_results = pd.DataFrame(columns=experiments.keys())
    em_results = pd.DataFrame(columns=experiments.keys())
    for experiment_name, experiment in experiments.items():
        em, f1 = zip(*list(map(
            lambda r: evaluate(r.correct, r.given), experiment.results
        )))
        em_results[experiment_name] = em
        f1_results[experiment_name] = f1

    os.makedirs("./results/", exist_ok=True)
    f1_results.to_csv("./results/f1_scores.csv")
    em_results.to_csv("./results/em_scores.csv")
