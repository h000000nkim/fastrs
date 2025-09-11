import os
import json
import urllib.request
import gzip
import shutil
import numpy as np
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from typing import Union, Any
from .exceptions import UtilError


__all__ = [
    "get_pretrained_model",
    "load_config",
    "load_color_schemes",
    "load_plot_config",
    "load_reduction_defaults",
    "load_fasttext_defaults",
]


def load_config(config_name: str) -> dict:
    """
    _config 폴더에서 JSON 설정 파일 로드

    Args:
        config_name: 설정 파일 이름 (확장자 제외)

    Returns:
        dict: 설정 딕셔너리
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "_config", f"{config_name}.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise UtilError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    except json.JSONDecodeError as e:
        raise UtilError(f"설정 파일을 파싱할 수 없습니다: {config_path}, 오류: {e}")


def load_color_schemes() -> dict:
    """색상 스키마 설정 로드"""
    return load_config("color_schemes")


def load_plot_config() -> dict:
    """플롯 설정 로드"""
    return load_config("plot_config")


def load_reduction_defaults() -> dict:
    """차원축소 알고리즘 기본값 설정 로드"""
    return load_config("reduction_defaults")


def load_fasttext_defaults() -> dict:
    """FastText 모델 기본값 설정 로드"""
    return load_config("fasttext_defaults")


def get_pretrained_model(
    url: str = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz",
    model_dir: str = "./models",
) -> FastText:
    """Download (if needed) and load Facebook FastText binary via gensim.

    Note: `FastText.load` expects a gensim-saved model. Facebook binaries
    must be loaded with `load_facebook_model` after decompressing the .gz.
    """
    os.makedirs(model_dir, exist_ok=True)
    bin_path = os.path.join(model_dir, "cc.ko.300.bin")
    gz_path = os.path.join(model_dir, "cc.ko.300.bin.gz")

    if os.path.exists(bin_path):
        return load_facebook_model(bin_path)

    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return load_facebook_model(bin_path)


def typecheck(input: Any | tuple[Any], expected: type | list[type]) -> None:
    if isinstance(input, tuple):
        if isinstance(expected, list):
            for each, exp in zip(input, expected):
                if isinstance(each, exp):
                    pass
                else:
                    raise TypeError(
                        f"Type of input must be {exp}, but got {type(each)}"
                    )
        else:
            for each in input:
                if isinstance(each, expected):
                    pass
                else:
                    raise TypeError(
                        f"Type of input must be {expected}, but got {type(each)}"
                    )
    else:
        if isinstance(expected, list):
            if any(isinstance(input, exp) for exp in expected):
                pass
            else:
                raise TypeError(
                    f"Type of input must be one of {expected}, but got {type(input)}"
                )
        elif isinstance(input, expected):
            pass
        else:
            raise TypeError(f"Type of input must be {expected}, but got {type(input)}")
    return None


def literalcheck(
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
    if answer is not None and response is not None and information is not None:
        pass
    else:
        raise UtilError(
            f"answer, response, and information must be provided. But answer: {type(answer)}, response: {type(response)}, information: {type(information)}"
        )
    if len(answer) == len(response) == (len(information)):
        pass
    else:
        raise UtilError(
            f"Length of answer, response, and information must be the same. But answer: {len(answer)}, response: {len(response)}, information: {len(information)}"
        )

    result = {}
    for i in range(len(answer)):
        key = f"item{i + 1}"
        result[key] = {
            "information": information[i],
            "answer": answer[i],
            "response": response[i],
        }
    return result


def validData(
    data: dict,
) -> None:
    for itemkey in data.keys():
        if isinstance(data[itemkey]["answer"], (list, tuple, np.ndarray)):
            if isinstance(data[itemkey]["answer"], np.ndarray):
                if data[itemkey]["answer"].ndim == 1:
                    pass
                else:
                    raise UtilError(
                        f"answer must be 1-dimensional numpy array, but for item `{itemkey}`: {data[itemkey]['answer'].ndim}D array"
                    )
        else:
            raise UtilError(
                f"answer must be list, tuple, or 1D numpy array, but for item `{itemkey}`: {type(data[itemkey]['answer'])}"
            )

        if isinstance(data[itemkey]["response"], (list, tuple, np.ndarray)):
            if isinstance(data[itemkey]["response"], np.ndarray):
                if data[itemkey]["response"].ndim == 1:
                    pass
                else:
                    raise UtilError(
                        f"response must be 1-dimensional numpy array, but for item `{itemkey}`: {data[itemkey]['response'].ndim}D array"
                    )
        else:
            raise UtilError(
                f"response must be list, tuple, or 1D numpy array, but for item `{itemkey}`: {type(data[itemkey]['response'])}"
            )

        if isinstance(data[itemkey]["information"], str):
            pass
        else:
            raise UtilError(
                f"information must be str, but for item `{itemkey}`: {type(data[itemkey]['information'])}"
            )

        if len(data[itemkey]["answer"]) > 0:
            pass
        else:
            raise UtilError(
                f"Length of answer must be greater than 0. But for item `{itemkey}`: {len(data[itemkey]['answer'])}"
            )

        if all(isinstance(x, str) for x in data[itemkey]["answer"]):
            pass
        else:
            raise UtilError(
                f"All elements in answer must be type str, but for item `{itemkey}`: {set(type(x) for x in data[itemkey]['answer'])}"
            )
        if all(isinstance(x, str) for x in data[itemkey]["response"]):
            pass
        else:
            raise UtilError(
                f"All elements in response must be type str, but for item `{itemkey}`: {set(type(x) for x in data[itemkey]['response'])}"
            )

    return None
