from .util import get_pretrained_model
from gensim.models import FastText


def fine_tune(
        item : list[list[str]],
        model : FastText = get_pretrained_model(), 
    ) -> FastText: 
    model.build_vocab(
        sentences=item, 
        update=True
        )
    model.train(
        corpus_iterable=item, 
        total_examples=len(item), 
        epochs=5
        )
    return model

def train(
        item : list[list[str]],
    ) -> FastText:
    pass
