from datasets import DatasetDict, load_dataset

from src.retrievers.fais_retriever import FAISRetriever
from src.utils.log import get_logger
from src.evaluation import evaluate
from typing import cast

logger = get_logger()


if __name__ == '__main__':
    dataset_name = "GroNLP/ik-nlp-22_slp"
    paragraphs = load_dataset(dataset_name, "paragraphs")
    questions = cast(DatasetDict, load_dataset(dataset_name, "questions"))

    questions_test = questions["test"]

    logger.info(questions)

    # Initialize retriever
    r = FAISRetriever()

    # # Retrieve example
    example_q = "What is the perplexity of a language model?"
    scores, result = r.retrieve(example_q)

    logger.info(
        f"Example q: {example_q} answer: {result['text'][0]}")

    for i, score in enumerate(scores):
        logger.info(f"Result {i+1} (score: {score:.02f}):")
        logger.info(result['text'][i])

    # Compute overall performance
    exact_match, f1_score = evaluate(
        r, questions_test["question"], questions_test["answer"])
    logger.info(f"Exact match: {exact_match:.02f}\n"
                f"F1-score: {f1_score:.02f}")
