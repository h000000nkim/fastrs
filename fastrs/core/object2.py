import os
import jamo
import numpy as np
import pandas as pd
from umap import UMAP
from typing import Union, Literal, Dict, Any, List, overload, TypedDict, Required, NotRequired, Unpack
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from . import util
from .preprocessor import Preprocessor
from .visualizer import visualize_embeddings


class Fastrs:


    @overload
    def __init__(
        self
    ) -> None:
        ...

    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]] = None,
        answers: np.ndarray = None,
        responses: np.ndarray = None,
        informations: np.ndarray = None,
        model: Union[FastText, None] = None,
    ):
        pass

    def preprocess(
        self,
        target: list[str] | None = None,
        space : Literal["single allow", "allow", "forbid1"] = "forbid",
        special: Literal["allow", "forbid"] = "forbid",
        unicode: Literal["allow", "forbid"] = "forbid",
        tab: Literal["allow", "forbid"] = "forbid",
        caps : Literal["allow", "forbid"] = "forbid",
        extra_deny: list[str] | None = None,
        extra_allow: list[str] | None = None
    ) -> None:
        """
        텍스트 정제
        """
        target = target if target is not None else [item.name for item in self.items]




        pass


class Item:

    def __init__(
        self,
        name: str,
        answer: List[str],
        response: List[str],
        information: str = None,
    ) -> None:
        """
        """
        self.name = name
        self.answer = answer
        self.response = response
        self.information = information
        pass

    def clean(
        self,
        space : Literal["single allow", "allow", "deny"] = "deny",
        special: Literal["allow", "deny"] = "deny",
        unicode: Literal["allow", "deny"] = "deny",
        tab: Literal["allow", "deny"] = "deny",
        caps : Literal["allow", "deny"] = "deny",
        extra_deny: list[str] = [],
        extra_allow: list[str] = []        
    ) -> None:
        """
        텍스트 정제
        """
        pass


#Parameter Options

class PreprocessParams(TypedDict, total=False):
    space: Literal["single allow", "allow", "deny"] = "deny"
    special: Literal["allow", "deny"] = "deny"
    unicode: Literal["allow", "deny"] = "deny"
    tab: Literal["allow", "deny"] = "deny"
    caps: Literal["allow", "deny"] = "deny"
    extra_deny: list[str] = []
    extra_allow: list[str] = []

class FastTextParams(TypedDict, total=False):
    vector_size: int
    window: int
    min_count: int
    workers: int
    sg: int
    hs: int
    negative: int
    ns_exponent: float
    epoch: int
    min_n: int
    max_n: int
    word_ngrams: int
    bucket: int
    lr: float
    lr_update_rate: int
    sample: float
    pretrained_vectors: str

class FineTuneParams(TypedDict, total=False):
    epochs: int