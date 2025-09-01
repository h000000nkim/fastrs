import jamo
import re
import regex
from pecab import PeCab
from typing import Union, Literal
import numpy as np

#clean
def clean(
        string: str,
        space : Literal["single allow", "allow", "deny"] = "deny",
        special: Literal["allow", "deny"] = "deny",
        unicode: Literal["allow", "deny"] = "deny",
        tab: Literal["allow", "deny"] = "deny",
        caps : Literal["allow", "deny"] = "deny",
        extra_deny: list[str] = [],
        extra_allow: list[str] = []
    ) -> str:
    string = "" if np.isnan(string) else str(string).strip()

    #extra
    for deny in extra_deny:
        string = string.replace(deny, '')

    allow_map = {}
    for idx, allow in enumerate(extra_allow):
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

#tokenize
def tokenize(string: str):
    tokenizer = PeCab()
    return tokenizer.morphs(string)

def jamoize(string: str):
    return jamo.h2hcj(jamo.h2j(string))


#format