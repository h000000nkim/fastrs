import os
import urllib.request
import gzip
import shutil
from gensim.models import FastText
from typing import Union, Literal


__all__ = [
    "get_pretrained_model"
]


def get_pretrained_model(
        url: str = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz",
        model_dir: str = "./models",
    ) -> FastText:
    bin_path = os.path.join(model_dir, 'cc.ko.300.bin')

    if os.path.exists(bin_path):
        return FastText.load(bin_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gz_path = os.path.join(model_dir, 'cc.ko.300.bin.gz')
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(bin_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return FastText.load(bin_path)


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