import jamo
import re
import regex
from pecab import PeCab
from typing import Union, Literal
import numpy as np


class Preprocessor:
    
    def __init__(
        self,
        answers: Union[str, list[str]],
        responses: list[str],
        information: Union[str, None] = None,
        option: Literal["automatic", "manual"] = "automatic",
    ) -> None:
        self.answer = answers if isinstance(answers, list) else [answers]
        self.originalanswer = self.answer
        self.response = responses
        self.originalresponse = self.response 
        self.information = information
        self.originalinformation = self.information
        if option == "automatic":
            self.preprocess("all")

    def preprocess(
            self,
            target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all", 
            space : Literal["single allow", "allow", "deny"] = "deny",
            special: Literal["allow", "deny"] = "deny",
            unicode: Literal["allow", "deny"] = "deny",
            tab: Literal["allow", "deny"] = "deny",
            caps : Literal["allow", "deny"] = "deny",
            extra_deny: list[str] = [],
            extra_allow: list[str] = []
    ) -> None:
        kwargs = dict(
            space=space,
            special=special,
            unicode=unicode,
            tab=tab,
            caps=caps,
            extra_deny=extra_deny,
            extra_allow=extra_allow
        )
        self.clean(target=target, **kwargs)
        self.tokenize(target=target)
        self.jamoize(target=target)        

        return None
    
    def clean(
            self,
            target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all", 
            space : Literal["single allow", "allow", "deny"] = "deny",
            special: Literal["allow", "deny"] = "deny",
            unicode: Literal["allow", "deny"] = "deny",
            tab: Literal["allow", "deny"] = "deny",
            caps : Literal["allow", "deny"] = "deny",
            extra_deny: list[str] = [],
            extra_allow: list[str] = []
    ) -> None:
        kwargs = dict(
            space=space,
            special=special,
            unicode=unicode,
            tab=tab,
            caps=caps,
            extra_deny=extra_deny,
            extra_allow=extra_allow
        )
        target = [target] if isinstance(target, str) else target
        if "answer" in target or "all" in target:
            self.cleananswer = [
                cleantxt(eanswer, **kwargs) 
                for eanswer in self.answer
            ]
            self.answer = self.cleananswer
        if "response" in target or "all" in target:
            self.cleanresponse = [
                cleantxt(eresponse, **kwargs) 
                for eresponse in self.response
            ]
            self.response = self.cleanresponse
        if ("information" in target or "all" in target) and self.information:
            self.cleaninformation = cleantxt(self.information, **kwargs)
            self.information = self.cleaninformation
        elif ("information" in target or "all" in target) and not self.information:
            return None
        
        else: raise ValueError("Invalid target")

    def tokenize(
            self,
            target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all"
    ) -> None:
        target = [target] if isinstance(target, str) else target
        if "answer" in target or "all" in target:
            self.tokenanswer = [token for eanswer in self.answer for token in tokenizetxt(eanswer)]
            self.answer = self.tokenanswer
            return self.tokenanswer 
        if "response" in target or "all" in target:
            self.tokenresponse = [token for eresponse in self.response for token in tokenizetxt(eresponse)]
            self.response = self.tokenresponse
            return self.tokenresponse
        if ("information" in target or "all" in target) and self.information:
            self.tokeninformation = tokenizetxt(self.information)
            self.information = self.tokeninformation
            return self.tokeninformation
        elif ("information" in target or "all" in target) and not self.information:
            return None
        else: raise ValueError("Invalid target")

    def jamoize(
            self,
            target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all"
    ) -> None:
        target = [target] if isinstance(target, str) else target
        if "answer" in target or "all" in target:
            self.jamoanswer = [jamoizetxt(eanswer) for eanswer in self.answer]
            return self.jamoanswer
        if "response" in target or "all" in target:
            self.jamoresponse = [jamoizetxt(eresponse) for eresponse in self.response]
            return self.jamoresponse
        if ("information" in target or "all" in target) and self.information:
            self.jamoinformation = jamoizetxt(self.information)
            return self.jamoinformation
        elif ("information" in target or "all" in target) and not self.information:
            return None
        else: raise ValueError("Invalid target")

    def format(
            self,
            information : bool = True,
    ) -> None:
        self.result = formattxt(
            answer=self.answer,
            response=self.response,
            information=self.information if information else None 
        )

#functions
def cleantxt(
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

    return string

def tokenizetxt(string: str):
    tokenizer = PeCab()
    return tokenizer.morphs(string)

def jamoizetxt(string: str):
    return jamo.h2hcj(jamo.h2j(string))

def formattxt(answer: list[str],
              response: list[str],
              information: Union[list[str], None]) -> list[str]:
    if information :
        return [information + [item] for item in answer + response]
    else:
        return answer + response