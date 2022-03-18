from src.fais_retriever import FAISRetriever
from src.utils.log import get_logger


logger = get_logger()


if __name__ == '__main__':
    # Initialize retriever
    r = FAISRetriever()

    # Retrieve example
    scores, result = r.retrieve(
        "What is the perplexity of a language model?")

    for i, score in enumerate(scores):
        logger.info(f"Result {i+1} (score: {score:.02f}):")
        logger.info(result['text'][i])

    # Compute overall performance
    exact_match, f1_score = r.evaluate()
    logger.info(f"Exact match: {exact_match:.02f}\n"
                f"F1-score: {f1_score:.02f}")
