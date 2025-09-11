import jamo
import re
import regex
import spacy
from pecab import PeCab
from typing import Union, Literal, Callable
import numpy as np
from . import util

__all__ = [
    "clean",
    "tokenize",
    "jamoize",
    "formatize"
]

koreantokenizer = PeCab()
try:
    englishtokenizer = spacy.load("en_core_web_sm")
except Exception:
    # Fallback to a blank English tokenizer if the small model is unavailable
    englishtokenizer = spacy.blank("en")

def clean(
    string: str,
    space : Literal["single allow", "allow", "forbid"] = "forbid",
    special: Literal["allow", "forbid"] = "forbid",
    unicode: Literal["allow", "forbid"] = "forbid",
    tab: Literal["allow", "forbid"] = "forbid",
    caps : Literal["allow", "forbid"] = "forbid",
    extra_forbid: list[str] = None,
    extra_allow: list[str] = None
) -> str:
    util.typecheck(string, str)
    
    # Validate literal parameters
    util.literalcheck(space, ["single allow", "allow", "forbid"])
    util.literalcheck(special, ["allow", "forbid"])
    util.literalcheck(unicode, ["allow", "forbid"])
    util.literalcheck(tab, ["allow", "forbid"])
    util.literalcheck(caps, ["allow", "forbid"])
    
    string = string.strip()

    #extra
    for forbid in extra_forbid if extra_forbid is not None else []:
        string = string.replace(forbid, '')

    allow_map = {}
    for idx, allow in enumerate(extra_allow if extra_allow is not None else []):
        replace_token = f"9allow{idx}9"  # Use pattern that won't be affected by case or special char removal
        allow_map[replace_token] = allow
        string = string.replace(allow, replace_token)
    
    #space
    if space == "single allow": string = re.sub(r'\s+', r' ', string)
    if space == "allow": pass
    if space == "forbid": string = re.sub(r'\s+', '', string)

    #special
    if special == "allow": pass
    elif special == "forbid": 
        string = regex.sub(r'\p{P}', '', string)
        string = regex.sub(r'\p{S}', '', string)
        if extra_allow and '_' in extra_allow:
            pass

    #unicode
    if unicode == "allow": pass
    if unicode == "forbid":
        RE_CF = regex.compile(r"\p{Cf}+")
        string = RE_CF.sub("", string)

    #tab
    if tab == "allow": pass
    if tab == "forbid": string = re.sub(r'\t', '', string)

    #caps
    if caps == "allow": pass
    if caps == "forbid": string = string.lower()

    #allowance recovery
    for token, original in allow_map.items():
        string = string.replace(token, original)

    #remove undervars
    string = re.sub(r'_', '', string)

    return string

def tokenize(
    string: str, 
    option: Literal["morphs", "nouns"] = "morphs"
) -> list[str]:
    util.typecheck(string, str)
    util.literalcheck(option, ["morphs", "nouns"])
    
    result = []

    if option == "morphs":
        tokens = koreantokenizer.morphs(string)
        for token in tokens:
            if re.match(r"^[A-Za-z]", token):
                doc = englishtokenizer(token)
                result.extend([t.text for t in doc])
            else:
                result.append(token)
    elif option == "nouns":
        tokens = koreantokenizer.nouns(string)
        for token in tokens:
            if re.match(r"^[A-Za-z]", token):
                doc = englishtokenizer(token)
                result.extend([t.text for t in doc])
            else:
                result.append(token)
    else:
        raise ValueError("function tokenize's option must be 'morphs' or 'nouns'")
    return result

def jamoize(
    string: str
) -> str:
    try:
        return jamo.j2hcj(jamo.h2j(string))
    except (TypeError, ValueError) as e:
        # If jamo conversion fails (e.g., non-Korean characters), return original string
        return string

def formatize(
    iterables : list[list[str]],
    anchor: list[str] | None = None,
    combine : bool = True
) -> list[list[str]]:
    util.typecheck(iterables, list)
    util.typecheck(combine, bool)
    if anchor is not None:
        util.typecheck(anchor, list)
    
    anchor = anchor if anchor is not None else []
    result = []
    for ls in iterables:
        if combine:
            result.extend(
                [anchor + [word] for word in ls]
            )
        else:
            result.append(ls)
    if not combine: result.append(anchor)
    return result
