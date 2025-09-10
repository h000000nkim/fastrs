import os
import jamo
import hdbscan
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from umap import UMAP
from typing import Union, Literal, Dict, Any, List, overload, TypedDict, Required, NotRequired, Unpack, Tuple, Optional
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from . import util
from . import preprocessor
from .exceptions import FastrsError, TrainingError, ReducerError, ItemError
from .visualizer import scatter


class Fastrs:

    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]] = None,
        answers: np.ndarray = None,
        responses: np.ndarray = None,
        informations: np.ndarray = None,
        model: Union[FastText, None] = None,
    ) -> None:
        self.model = model
        if data is not None : self.data = data
        else: 
            util.typecheck(answers, np.ndarray)
            util.typecheck(responses, np.ndarray)
            util.typecheck(informations, np.ndarray) if informations is not None else None
            self.data = util.formatData(answers, responses, informations)
        util.validData(self.data)
        self.items = [
            Item(
                name, 
                v['answer'], 
                v['response'],
                v.get('information', None)) 
                for name, v in self.data.items()
        ]

    def preprocess(
        self,
        option: Literal["default", "custom"] = "default"
    ) -> None:
        if option == "default":
            self.clean(target="information")
            self.tokenize(target="all", option="morphs")
            self.jamoize(target="all")
            self.formatize(iterables=["answer", "response"], anchor="information", combine=True)
            return self.feed
        elif option == "custom":
            raise NotImplementedError("Custom preprocessing not implemented yet.")

    def finetune(
        self,
        model: FastText = None,
    ) -> FastText:
        """
        """
        model = model or self.model or util.get_pretrained_model()
        model.build_vocab(self.feed, update=True, trim_rule=None)
        model.train(
            corpus_iterable=self.feed,
            total_examples=len(self.feed),
            epochs=self.epochs,
        )
        self.model = model
        return model
    
    def train(
        self,
        sg: int = 0,
        hs: int = 0,
        vector_size: int = 100,
        alpha: float = 0.025,
        window: int = 5,
        min_count: int = 5,
        max_vocab_size: Any | None = None,
        word_ngrams: int = 1,
        sample: float = 0.001,
        seed: int = 1,
        workers: int = 3,
        min_alpha: float = 0.0001,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_mean: int = 1,
        hashfxn: Any = hash,
        epochs: int = 5,
        null_word: int = 0,
        min_n: int = 3,
        max_n: int = 6,
        sorted_vocab: int = 1,
        bucket: int = 2000000,
        trim_rule: Any | None = None,
        callbacks: Any = (),
        max_final_vocab: Any | None = None,
        shrink_windows: bool = True
    ) -> FastText:
        fast_params = {
            "sg": sg,
            "hs": hs,
            "vector_size": vector_size,
            "alpha": alpha,
            "window": window,
            "min_count": min_count,
            "max_vocab_size": max_vocab_size,
            "word_ngrams": word_ngrams,
            "sample": sample,
            "seed": seed,
            "workers": workers,
            "min_alpha": min_alpha,
            "negative": negative,
            "ns_exponent": ns_exponent,
            "cbow_mean": cbow_mean,
            "hashfxn": hashfxn,
            "epochs": epochs,
            "null_word": null_word,
            "min_n": min_n,
            "max_n": max_n,
            "sorted_vocab": sorted_vocab,
            "bucket": bucket,
            "trim_rule": trim_rule,
            "callbacks": callbacks,
            "max_final_vocab": max_final_vocab,
            "shrink_windows": shrink_windows
        }
        if not hasattr(self, 'feed'): raise TrainingError("No preprocessed data found. Please run preprocess() or run formatize() before training.")
        self.model = FastText(
            sentences=self.feed,
            **fast_params
        )
        return self.model

    @overload
    def reduce(
        self,
        method: Literal["umap"] = "umap",
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: Optional[int] = None,
        n_components: int = 2,
        n_epochs: Optional[int] = None,
        init: Literal["spectral", "random"] = "spectral",
        **method_params: Any,
    ) -> pd.DataFrame: ...

    @overload
    def reduce(
        self,
        method: Literal["pca"],
        *,
        n_components: int = 2,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
        **method_params: Any,
    ) -> pd.DataFrame: ...

    @overload
    def reduce(
        self,
        method: Literal["tsne"],
        *,
        perplexity: float = 30.0,
        learning_rate: float | Literal["auto"] = "auto",
        n_iter: int = 1000,
        metric: str = "euclidean",
        init: Literal["random", "pca"] = "random",
        early_exaggeration: float = 12.0,
        angle: float = 0.5,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        n_components: int = 2,
        **method_params: Any,
    ) -> pd.DataFrame: ...

    def reduce(
        self,
        method: Literal["pca", "tsne", "umap"] = "umap",
        **method_params: Any,
    ) -> pd.DataFrame:
        method_params["n_components"] = 2 if "n_components" not in method_params else method_params["n_components"]
        if method == "umap":
            reducer = UMAP(**method_params)
        elif method == "pca":
            reducer = PCA(**method_params)
        elif method == "tsne":
            reducer = TSNE(**method_params)
        else:
            raise ValueError(f"Unknown method: {method}")
        reduced_vectors = reducer.fit_transform(self.model.wv.vectors)
        tokens = list(self.model.wv.key_to_index.keys())
        demension2_data = pd.DataFrame(reduced_vectors, columns=["x", "y"])
        token_data = pd.DataFrame(tokens, columns=["token"])
        result = pd.concat([demension2_data, token_data], axis=1)
        result["response"] = result["token"].apply(lambda x: self.jamodict.get(x, None))
        if result["response"].isna().any(): 
            raise ReducerError(
                f"Some tokens could not be mapped back to responses. {result[result['response'].isna()]['token'].tolist()}")
        result = result[["response", "token", "x", "y", "count"]]
        self.coordinates = result
        for item in self.items:
            item.coordinates = (
                result[result["response"].isin(item.original_answer + item.original_response)]
                .copy()
                .reset_index(drop=True)
            )
        return result

    def hdbscanize(
        self,
    ) -> None:
        if not hasattr(self, 'coordinates'):
            raise FastrsError("No reduced coordinates found. Please run reduce() before clustering.")
        labeled = []
        for item in self.items:
            item.hdbscanize()
            labeled.append(item.labels)
        self.labels = labeled

    def visualize(
        self
    ) -> list[go.Figure]:
        if not hasattr(self, 'coordinates'):
            raise FastrsError("No reduced coordinates found. Please run reduce() before visualization.")
        figs = []
        for item in self.items:
            item.visualize()
            figs.append(item.plot)
        self.plot = figs
        return figs

    def clean(
        self,
        item : list[str] | None = None,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all",
        space : Literal["single allow", "allow", "forbid1"] = "forbid",
        special: Literal["allow", "forbid"] = "forbid",
        unicode: Literal["allow", "forbid"] = "forbid",
        tab: Literal["allow", "forbid"] = "forbid",
        caps : Literal["allow", "forbid"] = "forbid",
        extra_forbid: list[str] | None = None,
        extra_allow: list[str] | None = None,
    ) -> None:
        """
        """
        util.literalcheck(target, ["all", "answer", "response", "information"])
        util.literalcheck(space, ["single allow", "allow", "forbid"])
        util.literalcheck(special, ["allow", "forbid"])
        util.literalcheck(unicode, ["allow", "forbid"])
        util.literalcheck(tab, ["allow", "forbid"])
        util.literalcheck(caps, ["allow", "forbid"])
        util.typecheck(extra_forbid, list) if extra_forbid is not None else None
        util.typecheck(extra_allow, list) if extra_allow is not None else None
        self.cleanparams = {
            "space": space,
            "special": special,
            "unicode": unicode,
            "tab": tab,
            "caps": caps,
            "extra_forbid": extra_forbid if extra_forbid is not None else [],
            "extra_allow": extra_allow if extra_allow is not None else []
        }
        item = item if item is not None else [eachitem.name for eachitem in self.items]
        target = [target] if isinstance(target, str) else target
        for i, current_item in enumerate(self.items):
            if current_item.name in item:
                current_item.clean(target=target, **self.cleanparams)
                self.items[i] = current_item
        return [it for it in self.items if it.name in item]
    
    def tokenize(
        self,
        item : list[str] | None = None,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all",
        option: Literal["morphs", "nouns"] = "morphs",
    ) -> None:
        """
        """
        item = item if item is not None else [eachitem.name for eachitem in self.items]
        target = [target] if isinstance(target, str) else target
        for i, current_item in enumerate(self.items):
            if current_item.name in item:
                current_item.tokenize(target=target, option=option)
                self.items[i] = current_item
        return [it for it in self.items if it.name in item]
    
    def jamoize(
        self,
        item : list[str] | None = None,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all"
    ) -> None:
        """
        """
        util.literalcheck(target, ["all", "answer", "response", "information"])
        item = item if item is not None else [eachitem.name for eachitem in self.items]
        target = [target] if isinstance(target, str) else target
        for i, current_item in enumerate(self.items):
            if current_item.name in item:
                current_item.jamoize(target=target)
                self.items[i] = current_item
        return [it for it in self.items if it.name in item]
    
    def formatize(
        self,
        iterables : list[Literal["answer", "response", "information"]],
        anchor : Literal["answer", "response", "information"],
        item : list[str] | None = None,
        combine : bool = True,
    ) -> None:
        """
        """
        util.literalcheck(anchor, ["answer", "response", "information"])
        util.literalcheck(iterables, ["answer", "response", "information"])
        item = item if item is not None else [eachitem.name for eachitem in self.items]
        feeds = []
        for i, current_item in enumerate(self.items):
            if current_item.name in item:
                current_item.formatize(iterables=iterables, anchor=anchor, combine=combine)
                feeds.extend(current_item.feed)
        self.feed = feeds
        return feeds
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
        util.typecheck(name, str)
        util.typecheck(answer, list)
        util.typecheck(response, list)
        util.typecheck(information, str) if information is not None else None
        self.name = name
        self.answer = answer
        self.response = response
        self.information = information
        self.original_answer = answer
        self.original_response = response
        self.original_information = information

    def clean(
        self,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all",
        space : Literal["single allow", "allow", "forbid"] = "forbid",
        special: Literal["allow", "forbid"] = "forbid",
        unicode: Literal["allow", "forbid"] = "forbid",
        tab: Literal["allow", "forbid"] = "forbid",
        caps : Literal["allow", "forbid"] = "forbid",
        extra_forbid: list[str] = None,
        extra_allow: list[str] = None
    ) -> Union[Tuple[Union[str, List[str]]], List[str], str]:
        util.literalcheck(target, ["all", "answer", "response", "information"])
        """
        clean item's texts(answer, response, information)
        """
        self.cleanparams = {
            "space": space,
            "special": special,
            "unicode": unicode,
            "tab": tab,
            "caps": caps,
            "extra_forbid": extra_forbid if extra_forbid is not None else [],
            "extra_allow": extra_allow if extra_allow is not None else []
        }
        self.answer, self.response, self.information = self._match_target(target, preprocessor.clean, **self.cleanparams).values()
        self.clean_answer = self.answer if self.answer != self.original_answer else None
        self.clean_response = self.response if self.response != self.original_response else None
        self.clean_information = self.information if self.information != self.original_information else None
        return self._parse_return(target)

    def tokenize(
        self,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all",
        option: Literal["morphs", "nouns"] = "morphs",
    ) -> Tuple[list[list[str]] | list[str] | str, ...]:
        """
        tokenize item's texts(answer, response, information)
        """
        util.literalcheck(target, ["all", "answer", "response", "information"])
        util.literalcheck(option, ["morphs", "nouns"])
        self.answer, self.response, self.information = self._match_target(target, preprocessor.tokenize, option=option).values()
        self.token_answer = self.answer if self.answer != (self.clean_answer if self.clean_answer is not None else self.original_answer) else None
        self.token_response = self.response if self.response != (self.clean_response if self.clean_response is not None else self.original_response) else None
        self.token_information = self.information if self.information != (self.clean_information if self.clean_information is not None else self.original_information) else None
        return self._parse_return(target)

    def jamoize(
        self,
        target: Literal["all", "answer", "response", "information"] | list[Literal["answer", "response", "information"]] = "all"
    ) -> Tuple[list[list[str]] | list[str] | str, ...]:
        """
        jamoize item's texts(answer, response, information)
        """
        util.literalcheck(target, ["all", "answer", "response", "information"])
        ansresps = self.original_answer + self.original_response
        self.answer, self.response, self.information = self._match_target(target, preprocessor.jamoize).values()
        jamos = self.answer + self.response
        self.jamodict = {orig: jamoed for orig, jamoed in zip(ansresps, jamos)}
        self.jamoized_answer = self.answer if self.answer != (self.token_answer if self.token_answer is not None else (self.clean_answer if self.clean_answer is not None else self.original_answer)) else None
        self.jamoized_response = self.response if self.response != (self.token_response if self.token_response is not None else (self.clean_response if self.clean_response is not None else self.original_response)) else None
        self.jamoized_information = self.information if self.information != (self.token_information if self.token_information is not None else (self.clean_information if self.clean_information is not None else self.original_information)) else None
        return self._parse_return(target)
    
    def formatize(
        self,
        iterables : list[Literal["answer", "response", "information"]],
        anchor : Literal["answer", "response", "information"],
        combine : bool = True,
    ) -> list[str]:
        util.literalcheck(anchor, ["answer", "response", "information"])
        util.literalcheck(iterables, ["answer", "response", "information"])
        """
        """
        anchor = [val for key, val in zip(["answer", "response", "information"], [self.answer, self.response, self.information])
                 if key in iterables and (val is not None)]
        anchor = sum([v if isinstance(v, list) else [v] for v in anchor], [])
        iterable = []
        if "answer" in iterables:
            iterable.append(self.answer)
        if "response" in iterables:
            iterable.append(self.response)
        if "information" in iterables:
            iterable.append(self.information)
        self.feed = preprocessor.formatize(iterables=iterable, anchor=anchor, combine=combine)
        return self.feed
    
    def countize(
        self
    ) -> pd.DataFrame:
        """
        """
        all_texts = self.original_answer + self.original_response
        self.coordinates["count"] = self.coordinates["response"].apply(lambda x: all_texts.count(x))
        return self.coordinates

    def hdbscanize(
        self,
        **hdbscan_params: Dict[str, Any]
    ) -> pd.DataFrame:
        data = self.coordinates[["x", "y"]].values
        self.labels = self.coordinates
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size= len(self.original_answer) * 2 if hdbscan_params.get("min_cluster_size") is None else hdbscan_params.get("min_cluster_size"),
            **hdbscan_params
        )
        labels = clusterer.fit_predict(data)
        self.labels["label"] = labels
        return self.labels

    def visualize(
        self
    )-> go.Figure:
        """
        """
        if not (hasattr(self, 'coordinates') or hasattr(self, 'labels')):
            raise ItemError("No coordinates found. Please run reduce() before visualize().")
        self.plot = scatter(
            self.coordinates if not hasattr(self, 'labels') else self.labels,
            answers=self.original_answer,
            title=self.name,
            scatter_type="simple"
        )
        return self.plot

    def _match_target(self, target, func, **kwargs):
        target = [target] if isinstance(target, str) else target
        results = {}
        if "all" in target or "answer" in target:
            # Handle nested list structure for tokenized data
            if all(isinstance(item, list) for item in self.answer):
                # Flatten and apply function to each token
                results["answer"] = [func(token, **kwargs) for sublist in self.answer for token in sublist]
            else:
                results["answer"] = [func(s, **kwargs) for s in self.answer]
        else: results["answer"] = self.answer
        if "all" in target or "response" in target:
            # Handle nested list structure for tokenized data
            if all(isinstance(item, list) for item in self.response):
                # Flatten and apply function to each token
                results["response"] = [func(token, **kwargs) for sublist in self.response for token in sublist]
            else:
                results["response"] = [func(s, **kwargs) for s in self.response]
        else: results["response"] = self.response
        if "all" in target or "information" in target and self.information is not None:
            results["information"] = func(self.information, **kwargs)
        else: results["information"] = self.information
        return results
    
    def _parse_return(self, target):
        target = [target] if isinstance(target, str) else target
        result = []
        if "answer" in target: result.append(self.answer)
        if "response" in target: result.append(self.response)
        if "information" in target: result.append(self.information)
        if "all" in target: result = (self.answer, self.response, self.information)
        return tuple(result)