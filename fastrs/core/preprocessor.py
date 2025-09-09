import jamo
import re
import regex
import spacy
from pecab import PeCab
from typing import Union, Literal, Callable
import numpy as np

__all__ = [
    "clean",
    "tokenize",
    "jamoize",
    "formatize"
]

koreantokenizer = PeCab()
englishtokenizer = spacy.load("en_core_web_sm")

def clean(
    string: str,
    space : Literal["single allow", "allow", "deny"] = "deny",
    special: Literal["allow", "deny"] = "deny",
    unicode: Literal["allow", "deny"] = "deny",
    tab: Literal["allow", "deny"] = "deny",
    caps : Literal["allow", "deny"] = "deny",
    extra_deny: list[str] = None,
    extra_allow: list[str] = None
) -> str:
    string = "" if np.isnan(string) else str(string).strip()

    #extra
    for deny in extra_deny if extra_deny is not None else []:
        string = string.replace(deny, '')

    allow_map = {}
    for idx, allow in enumerate(extra_allow if extra_allow is not None else []):
        replace_token = f"__ALLOW{idx}__"
        allow_map[replace_token] = allow
        string = string.replace(allow, replace_token)
    
    #space
    if space == "single allow": string = re.sub(r'\s+', r' ', string)
    if space == "allow": pass
    if space == "deny": string = re.sub(r'\s+', '', string)

    #special <- not completed
    if special == "allow": pass
    elif special == "deny": 
        string = regex.sub(r'[\p{P}&&[^_]]+', '', string)

    #unicode
    if unicode == "allow": pass
    if unicode == "deny":
        RE_CF = regex.compile(r"\p{Cf}+")
        string = RE_CF.sub("", string)

    #tab
    if tab == "allow": pass
    if tab == "deny": string = re.sub(r'\t', '', string)

    #caps
    if caps == "allow": pass
    if caps == "deny": string = string.lower()

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
    return jamo.h2hcj(jamo.h2j(string))

def formatize(
    iterables : list[list[str]],
    anchor: list[str] | None = None,
    combine : bool = True
) -> list[list[str]]:
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