from transformers import DPRReader, DPRReaderTokenizer
from typing import List, Dict, Tuple


class DprReader():
    def __init__(self) -> None:
        self._tokenizer = DPRReaderTokenizer.from_pretrained(
            "facebook/dpr-reader-single-nq-base")
        self._model = DPRReader.from_pretrained(
            "facebook/dpr-reader-single-nq-base"
        )

    def read(self, query: str, context: Dict[str, List[str]]) -> List[Tuple]:
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
            encoded_inputs, outputs)

        return predicted_spans
