import os
import urllib.request
import gzip
import shutil
import inspect
import numpy as np
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from typing import Union, Literal
from .exceptions import PreprocessingError, TrainingError, UtilError



__all__ = [
    "get_pretrained_model"
]


def get_pretrained_model(
        url: str = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz",
        model_dir: str = "./models",
    ) -> FastText:
    """Download (if needed) and load Facebook FastText binary via gensim.

    Note: `FastText.load` expects a gensim-saved model. Facebook binaries
    must be loaded with `load_facebook_model` after decompressing the .gz.
    """
    os.makedirs(model_dir, exist_ok=True)
    bin_path = os.path.join(model_dir, 'cc.ko.300.bin')
    gz_path = os.path.join(model_dir, 'cc.ko.300.bin.gz')

    # If already decompressed, load directly
    if os.path.exists(bin_path):
        return load_facebook_model(bin_path)

    # Download compressed file if missing
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(url, gz_path)

    # Decompress to .bin
    with gzip.open(gz_path, 'rb') as f_in, open(bin_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return load_facebook_model(bin_path)


def _typecheck():
    pass


def _literalcheck(
        input: Union[str, list[str]],
        literal: list[str],
    ) -> None:
    if isinstance(input, str):
        input = [input]

    for item in input:
        if item not in literal:
            raise ValueError(f"Invalid input: {item}. Expected one of: {literal}")
        


def formatData(
        answer: np.ndarray,
        response: np.ndarray,
        information: np.ndarray = None,
) -> dict:
    
    #assertion
    if answer is not None and response is not None and information is not None: pass
    else: raise UtilError(
        f"answer, response, and information must be provided. But answer: {type(answer)}, response: {type(response)}, information: {type(information)}"
    )
    if len(answer) == len(response) == (len(information)): pass
    else: raise UtilError(
        f"Length of answer, response, and information must be the same. But answer: {len(answer)}, response: {len(response)}, information: {len(information)}"
    )

    #format
    result = {}
    for i in range(len(answer)):
        key = f"item{i+1}"
        result[key] = {
            "information": information[i],
            "answer": answer[i],
            "response": response[i]
        }
    return result

def validData(
        data: dict,
) -> None:
    for itemkey in data.keys():

        # type check for container types
        if isinstance(data[itemkey]["answer"], (list, tuple, np.ndarray)):
            if isinstance(data[itemkey]["answer"], np.ndarray):
                if data[itemkey]["answer"].ndim == 1: pass
                else: raise UtilError(
                    f"answer must be 1-dimensional numpy array, but for item `{itemkey}`: {data[itemkey]['answer'].ndim}D array"
                )
        else: raise UtilError(
            f"answer must be list, tuple, or 1D numpy array, but for item `{itemkey}`: {type(data[itemkey]['answer'])}"
        )
        
        if isinstance(data[itemkey]["response"], (list, tuple, np.ndarray)):
            if isinstance(data[itemkey]["response"], np.ndarray):
                if data[itemkey]["response"].ndim == 1: pass
                else: raise UtilError(
                    f"response must be 1-dimensional numpy array, but for item `{itemkey}`: {data[itemkey]['response'].ndim}D array"
                )
        else: raise UtilError(
            f"response must be list, tuple, or 1D numpy array, but for item `{itemkey}`: {type(data[itemkey]['response'])}"
        )
        
        if isinstance(data[itemkey]["information"], str): pass
        else: raise UtilError(
            f"information must be str, but for item `{itemkey}`: {type(data[itemkey]['information'])}"
        )

        #length check
        if (len(data[itemkey]["answer"]) ==
            len(data[itemkey]["response"])
        ): pass
        else: raise UtilError(
            f"Length of answer and response must be the same. But for item `{itemkey}` answer: {len(data[itemkey]['answer'])}, response: {len(data[itemkey]['response'])}")
        if len(data[itemkey]["answer"]) > 0: pass
        else: raise UtilError(
            f"Length of answer must be greater than 0. But for item `{itemkey}`: {len(data[itemkey]['answer'])}"
            )
        
        #type check for elements
        if all(isinstance(x, str) for x in data[itemkey]['answer']): pass
        else: raise UtilError(
            f"All elements in answer must be type str, but for item `{itemkey}`: {set(type(x) for x in data[itemkey]['answer'])}"
        )
        if all(isinstance(x, str) for x in data[itemkey]['response']): pass
        else: raise UtilError(
            f"All elements in response must be type str, but for item `{itemkey}`: {set(type(x) for x in data[itemkey]['response'])}"
        )

    return None

def copysignature(from_func):
    def decorator(to_func):
        to_func.__signature__ = inspect.signature(from_func)
        to_func.__doc__ = from_func.__doc__
        return to_func
    return decorator
