from typing import Any, Callable, List
from src.retrievers.base_retriever import Retriever

from src.utils.string_utils import (lower, remove_articles, remove_punc,
                                    white_space_fix)


def _normalize_text(inp: str, preprocessing_functions: List[Callable[[str], str]]):
    for fun in preprocessing_functions:
        inp = fun(inp)
    return inp


def _normalize_text_default(inp: str) -> str:
    """Preprocesses the sentence string by normalizing.

    Args:
        s (str): the sentence

    Returns:
        string: normalized with default parames
    """

    steps = [remove_articles, white_space_fix, remove_punc, lower]

    return _normalize_text(inp, steps)


def exact_match(prediction: str, answer: str) -> int:
    """Computes exact match for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        int: 1 for exact match, 0 for not
    """
    return int(_normalize_text_default(prediction) == _normalize_text_default(answer))


def f1(prediction: str, answer: str) -> float:
    """Computes F1-score on token overlap for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        boolean: the f1 score
    """
    pred_tokens = _normalize_text_default(prediction).split()
    answer_tokens = _normalize_text_default(answer).split()

    if len(pred_tokens) == 0 or len(answer_tokens) == 0:
        return int(pred_tokens == answer_tokens)

    common_tokens = set(pred_tokens) & set(answer_tokens)

    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(answer_tokens)

    return 2 * (prec * rec) / (prec + rec)


def evaluate(retriever: Retriever, questions: Any, answers: Any):
    """Evaluates the entire model by computing F1-score and exact match on the
    entire dataset.

    Returns:
        float: overall exact match
        float: overall F1-score
    """

    predictions = []
    scores = 0

    # Currently just takes the first answer and does not look at scores yet
    for question in questions:
        score, result = retriever.retrieve(question, 1)
        scores += score[0]
        predictions.append(result['text'][0])

    exact_matches = [exact_match(
        predictions[i], answers[i]) for i in range(len(answers))]
    f1_scores = [f1(
        predictions[i], answers[i]) for i in range(len(answers))]

    return sum(exact_matches) / len(exact_matches), sum(f1_scores) / len(f1_scores)
