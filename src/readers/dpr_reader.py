from transformers import DPRReader, DPRReaderTokenizer
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from src.readers.base_reader import Reader


load_dotenv()


class DprReader(Reader):
    def __init__(self) -> None:
        self._tokenizer = DPRReaderTokenizer.from_pretrained(
            "facebook/dpr-reader-single-nq-base")
        self._model = DPRReader.from_pretrained(
            "facebook/dpr-reader-single-nq-base")

    def read(self,
             query: str,
             context: Dict[str, List[str]],
             num_answers=5) -> List[Tuple]:
        encoded_inputs = self._tokenizer(
            questions=query,
            titles=context['titles'],
            texts=context['texts'],
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        outputs = self._model(**encoded_inputs)

        predicted_spans = self._tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            num_spans=num_answers,
            num_spans_per_passage=2,
            max_answer_length=512)

        return predicted_spans
