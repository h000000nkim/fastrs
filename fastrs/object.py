from gensim.models import FastText
from typing import Union, Literal, Dict, Any
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from . import util
import os
from .preprocessor import Preprocessor
import inspect

class Fastrs:
    @util.copysignature(FastText.__init__)
    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]] = None,
        answers: np.ndarray = None,
        responses: np.ndarray = None,
        informations: np.ndarray = None,
        model: Union[FastText, None] = None,
        option: Literal["automatic", "manual"] = "automatic",
        **params: Dict[str, Any],
    ) -> None:
        """
        """

        #set Attr
        self.model = model
        if data is not None : self.data = data
        else: self.data = util.formatData(answers, responses, informations)
        util.validData(self.data)

        if option == "automatic":
            traindata = []
            for item in self.data.values():
                preprocessed = Preprocessor(
                    answers=item["answer"],
                    responses=item["response"],
                    information=item.get("information", None),
                    option="automatic"
                )
                traindata.append(preprocessed)
            self.traindata = traindata
            self.sentences = [sent for pp in self.traindata for sent in pp]
            self.finetune(model if model is not None else util.get_pretrained_model())
        elif option == "manual": pass

    def finetune(
            self,
            model: FastText = None
    ) -> FastText:
        model = model or self.model or util.get_pretrained_model()
        model.build_vocab(self.sentences, update=True, trim_rule=None)
        model.train(
            corpus_iterable=self.sentences,
            total_examples=len(self.sentences),
            epochs=self.epochs,
        )
        self.model = model
        return model

    def train(
            self,
    ) -> FastText:
        model = FastText(
            sentences=self.sentences,
            **self.fasttext_params
        )
        self.model = model
        return model
    
    def reduce(
        self,
        method : Literal["umap", "pca", "tsne"] = "umap",
        **method_params: Dict[str, Any],
    ) -> pd.DataFrame:
        if method == "umap":
            reducer = UMAP(n_components=2, **method_params)
        elif method == "pca":
            reducer = PCA(n_components=2, **method_params)
        elif method == "tsne":
            reducer = TSNE(n_components=2, **method_params)
        else:
            raise ValueError(f"Unknown method: {method}")
        reduced_data = reducer.fit_transform(self.model.wv.vectors)
        return pd.DataFrame(reduced_data, columns=["x", "y"])

    def visualize(
        self,

    ) -> None:
        pass
