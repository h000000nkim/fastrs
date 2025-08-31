from gensim.models import FastText
from typing import Union, Literal


class Fastrs:
    
    def __init__(
            self,
            answer: Union[str, list[str]],
            response: list[str],
            information: str,
            model: Union[FastText, None] = None,
            ) -> None:
        pass

    def train(self) -> FastText:
        pass

    def reduce(self) -> None:
        pass

    