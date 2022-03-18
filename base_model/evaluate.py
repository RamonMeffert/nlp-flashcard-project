from typing import Callable, List

from base_model.string_utils import lower, remove_articles, remove_punc, white_space_fix


def normalize_text(inp: str, functions: List[Callable[[str], str]]):
    for fun in functions:
        inp = fun(inp)
    return inp


def normalize_text_default(inp: str) -> str:
    """Preprocesses the sentence string by normalizing.

    Args:
        s (str): the sentence

    Returns:
        string: normalized with default parames
    """

    steps = [remove_articles, white_space_fix, remove_punc, lower]

    return normalize_text(inp, steps)


def compute_exact_match(prediction: str, answer: str) -> int:
    """Computes exact match for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        int: 1 for exact match, 0 for not
    """
    return int(normalize_text_default(prediction) == normalize_text_default(answer))


def compute_f1(prediction: str, answer: str) -> float:
    """Computes F1-score on token overlap for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        boolean: the f1 score
    """
    pred_tokens = normalize_text_default(prediction).split()
    answer_tokens = normalize_text_default(answer).split()

    if len(pred_tokens) == 0 or len(answer_tokens) == 0:
        return int(pred_tokens == answer_tokens)

    common_tokens = set(pred_tokens) & set(answer_tokens)

    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(answer_tokens)

    return 2 * (prec * rec) / (prec + rec)
