import torch
from transformers import (
    LongformerTokenizerFast,
    LongformerForQuestionAnswering
)
from typing import List, Dict, Tuple

from src.readers.base_reader import Reader


class LongformerReader(Reader):
    def __init__(self) -> None:
        checkpoint = "valhalla/longformer-base-4096-finetuned-squadv1"
        self.tokenizer = LongformerTokenizerFast.from_pretrained(checkpoint)
        self.model = LongformerForQuestionAnswering.from_pretrained(checkpoint)

    def read(self,
             query: str,
             context: Dict[str, List[str]],
             num_answers=5) -> List[Tuple]:
        answers = []

        for text in context['texts']:
            encoding = self.tokenizer(
                query, text, return_tensors="pt")
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            outputs = self.model(input_ids, attention_mask=attention_mask)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[0].tolist())
            answer_tokens = all_tokens[
                torch.argmax(start_logits):torch.argmax(end_logits) + 1]
            answer = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(answer_tokens)
            )
            answers.append([answer, [], []])

        return answers
