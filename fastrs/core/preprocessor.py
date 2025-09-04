import jamo
import re
import regex
from pecab import PeCab
from typing import Union, Literal, Callable
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
        self.originalanswer = self.answer.copy()
        self.response = responses
        self.originalresponse = self.response.copy()
        self.information = information
        self.originalinformation = self.information
        if option == "automatic": 
            self.preprocess("all")

    def preprocess(self, target="all", **kwargs) -> None:
        """전체 전처리 파이프라인 실행"""
        self.clean(target=target, **kwargs)
        self.tokenize(target=target)
        self.jamoize(target=target)
        self.format(information=bool(self.information))

    def clean(self, target="all", **kwargs) -> None:
        """텍스트 클리닝"""
        targets = self._parse_target(target)
        self._apply_to_targets(targets, cleantxt, **kwargs)

    def tokenize(self, target="all") -> None:
        """토크나이징"""
        targets = self._parse_target(target)
        self._apply_to_targets(targets, tokenizetxt)

    def jamoize(self, target="all") -> None:
        """자모 분리"""
        targets = self._parse_target(target)
        self._apply_to_targets(targets, jamoizetxt)

    def format(
            self,
            information : bool = True,
    ) -> None:
        self.result = _formattxt_sentences(
            answer=self.answer,
            response=self.response,
            information=self.information if information else None 
        )
    
    def __iter__(self):
        if hasattr(self, 'result'):
            return iter(self.result)
        return iter(())

    def __len__(self):
        if hasattr(self, 'result'):
            return len(self.result)
        return 0

    def __getitem__(self, index):
        if hasattr(self, 'result'):
            return self.result[index]
        raise IndexError(index)
    
    def __repr__(self):
        if hasattr(self, 'result'):
            return repr(self.result)
        return f"Preprocessor(answers={len(self.answer)}, responses={len(self.response)})"

    def __getattr__(self, name):
        if hasattr(self, 'result'):
            return getattr(self.result, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _parse_target(self, target) -> list[str]:
        """target을 리스트로 변환"""
        target = [target] if isinstance(target, str) else target
        if "all" in target:
            return ["answer", "response", "information"]
        return target

    def _apply_to_targets(self, targets: list[str], func: Callable, **kwargs):
        """지정된 타겟들에 함수를 적용"""
        for target in targets:
            if target == "answer":
                if hasattr(self, 'answer'):
                    if target == "answer" and func == cleantxt:
                        self.cleananswer = [func(item, **kwargs) for item in self.answer]
                        self.answer = self.cleananswer
                    elif target == "answer" and func == tokenizetxt:
                        self.tokenanswer = [tokenizetxt(item) for item in self.answer]
                        self.answer = self.tokenanswer
                    elif target == "answer" and func == jamoizetxt:
                        if self.answer and isinstance(self.answer[0], list):
                            self.jamoanswer = [[jamoizetxt(tok) for tok in sent] for sent in self.answer]
                        else:
                            self.jamoanswer = [jamoizetxt(item) for item in self.answer]
                        
            elif target == "response":
                if hasattr(self, 'response'):
                    if target == "response" and func == cleantxt:
                        self.cleanresponse = [func(item, **kwargs) for item in self.response]
                        self.response = self.cleanresponse
                    elif target == "response" and func == tokenizetxt:
                        self.tokenresponse = [tokenizetxt(item) for item in self.response]
                        self.response = self.tokenresponse
                    elif target == "response" and func == jamoizetxt:
                        if self.response and isinstance(self.response[0], list):
                            self.jamoresponse = [[jamoizetxt(tok) for tok in sent] for sent in self.response]
                        else:
                            self.jamoresponse = [jamoizetxt(item) for item in self.response]
                        
            elif target == "information" and self.information:
                if func == cleantxt:
                    self.cleaninformation = func(self.information, **kwargs)
                    self.information = self.cleaninformation
                elif func == tokenizetxt:
                    self.tokeninformation = tokenizetxt(self.information)
                    self.information = self.tokeninformation
                elif func == jamoizetxt:
                    if isinstance(self.information, list):
                        self.jamoinformation = [jamoizetxt(tok) for tok in self.information]
                        self.information = self.jamoinformation
                    else:
                        self.jamoinformation = jamoizetxt(self.information)
                        self.information = self.jamoinformation

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
              information: Union[str, list[str], None]) -> list[str]:
    if information:
        if isinstance(information, str):
            # information이 str이면 각 item 앞에 추가
            return [information + " " + item for item in answer + response]
        else:
            # information이 list면 전체를 앞에 추가
            return information + answer + response
    else:
        return answer + response

def _formattxt_sentences(
    answer: list[Union[str, list[str]]],
    response: list[Union[str, list[str]]],
    information: Union[str, list[str], None]
) -> list[list[str]]:
    """Combine answers/responses into sentence-token lists and optionally prefix info tokens."""
    ans_sents = [a if isinstance(a, list) else [a] for a in answer]
    res_sents = [r if isinstance(r, list) else [r] for r in response]
    sentences = ans_sents + res_sents
    if information:
        info_tokens = information if isinstance(information, list) else [information]
        sentences = [info_tokens + sent for sent in sentences]
    return sentences
