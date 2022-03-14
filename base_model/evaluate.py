def normalize_text(s: str) -> str:
    """Preprocesses the sentence string by normalizing.

    Args:
        s (str): the sentence

    Returns:
        string: normalized sentence
    """
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, answer: str) -> int:
    """Computes exact match for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        int: 1 for exact match, 0 for not
    """
    return int(normalize_text(prediction) == normalize_text(answer))


def compute_f1(prediction: str, answer: str) -> float:
    """Computes F1-score on token overlap for sentences.

    Args:
        prediction (str): the predicted answer
        answer (str): the gold answer

    Returns:
        boolean: the f1 score
    """
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()

    if len(pred_tokens) == 0 or len(answer_tokens) == 0:
        return int(pred_tokens == answer_tokens)

    common_tokens = set(pred_tokens) & set(answer_tokens)

    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(answer_tokens)

    return 2 * (prec * rec) / (prec + rec)
