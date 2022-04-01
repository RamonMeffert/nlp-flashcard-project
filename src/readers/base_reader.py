from typing import Dict, List, Tuple


class Reader():
    def read(self,
             query: str,
             context: Dict[str, List[str]],
             num_answers: int) -> List[Tuple]:
        raise NotImplementedError()
