from gensim.models import FastText
from typing import Union, Literal, Dict, Any
from .preprocessor import cleantxt, tokenizetxt, jamoizetxt, formattxt
import numpy as np


class Fastrs:

    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]] = None,
        answers: np.ndarray = None,
        responses: np.ndarray = None,
        informations: np.ndarray = None,
        model: Union[FastText, None] = None,
        option: Literal["automatic", "manual"] = "automatic",
    ) -> None:
        """
        data가 있으면 data 그대로 사용, 없으면 answers/responses/informations로 data 형태 생성
        """
        # 타입 체크 및 형태 변환
        if data is not None:
            # data가 있는 경우: 그대로 사용
            self.data = data
        elif answers is not None and responses is not None:
            # data가 없고 개별 파라미터가 있는 경우: data 형태로 변환
            self.data = self._convert_arrays_to_data(answers, responses, informations)
        else:
            raise ValueError("data 또는 answers/responses 파라미터 중 하나는 반드시 제공되어야 합니다.")
        
        # 길이 체크
        self._validate_data_structure()
        
        # 원본 데이터 백업
        self.original_data = {key: value.copy() for key, value in self.data.items()}
        self.model = model
        
        if option == "automatic":
            self.preprocess("all")

    def _convert_arrays_to_data(
        self, 
        answers: np.ndarray, 
        responses: np.ndarray, 
        informations: np.ndarray = None
    ) -> Dict[str, Dict[str, Any]]:
        """numpy 배열들을 data 딕셔너리 형태로 변환"""
        
        # 길이 체크
        if len(answers) != len(responses):
            raise ValueError(f"answers와 responses의 길이가 다릅니다: {len(answers)} != {len(responses)}")
        
        if informations is not None and len(informations) != len(answers):
            raise ValueError(f"informations의 길이가 answers와 다릅니다: {len(informations)} != {len(answers)}")
        
        data = {}
        
        for i in range(len(answers)):
            item_id = f"item_{i}"
            
            # answers 처리
            answer_item = answers[i]
            if isinstance(answer_item, np.ndarray):
                answer_list = answer_item.tolist()
            elif isinstance(answer_item, (list, tuple)):
                answer_list = list(answer_item)
            else:
                answer_list = [str(answer_item)]
            
            # responses 처리
            response_item = responses[i]
            if isinstance(response_item, np.ndarray):
                response_list = response_item.tolist()
            elif isinstance(response_item, (list, tuple)):
                response_list = list(response_item)
            else:
                response_list = [str(response_item)]
            
            # informations 처리
            info_item = None
            if informations is not None:
                info_item = informations[i]
                if isinstance(info_item, np.ndarray):
                    if info_item.size == 1:
                        info_item = str(info_item.item())
                    else:
                        info_item = info_item.tolist()
                elif info_item is not None:
                    info_item = str(info_item)
            
            data[item_id] = {
                "answers": answer_list,
                "responses": response_list,
                "information": info_item
            }
        
        return data

    def _validate_data_structure(self):
        """data 구조 유효성 검사"""
        if not isinstance(self.data, dict):
            raise TypeError("data는 딕셔너리 형태여야 합니다.")
        
        for item_id, item_data in self.data.items():
            if not isinstance(item_data, dict):
                raise TypeError(f"data['{item_id}']는 딕셔너리 형태여야 합니다.")
            
            # 필수 키 체크
            if "answers" not in item_data or "responses" not in item_data:
                raise KeyError(f"data['{item_id}']에 'answers'와 'responses' 키가 모두 있어야 합니다.")
            
            # answers/responses가 리스트인지 체크 (문자열이면 리스트로 변환)
            if isinstance(item_data["answers"], str):
                self.data[item_id]["answers"] = [item_data["answers"]]
            elif not isinstance(item_data["answers"], list):
                raise TypeError(f"data['{item_id}']['answers']는 문자열 또는 리스트여야 합니다.")
            
            if isinstance(item_data["responses"], str):
                self.data[item_id]["responses"] = [item_data["responses"]]
            elif not isinstance(item_data["responses"], list):
                raise TypeError(f"data['{item_id}']['responses']는 문자열 또는 리스트여야 합니다.")

    def preprocess(
        self,
        target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all", 
        space: Literal["single allow", "allow", "deny"] = "deny",
        special: Literal["allow", "deny"] = "deny",
        unicode: Literal["allow", "deny"] = "deny",
        tab: Literal["allow", "deny"] = "deny",
        caps: Literal["allow", "deny"] = "deny",
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
        space: Literal["single allow", "allow", "deny"] = "deny",
        special: Literal["allow", "deny"] = "deny",
        unicode: Literal["allow", "deny"] = "deny",
        tab: Literal["allow", "deny"] = "deny",
        caps: Literal["allow", "deny"] = "deny",
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
        
        for item_id, item_data in self.data.items():
            if "answer" in target or "all" in target:
                if "answers" in item_data:
                    answers = item_data["answers"] if isinstance(item_data["answers"], list) else [item_data["answers"]]
                    self.data[item_id]["clean_answers"] = [cleantxt(answer, **kwargs) for answer in answers]
                    self.data[item_id]["answers"] = self.data[item_id]["clean_answers"]
                    
            if "response" in target or "all" in target:
                if "responses" in item_data:
                    responses = item_data["responses"] if isinstance(item_data["responses"], list) else [item_data["responses"]]
                    self.data[item_id]["clean_responses"] = [cleantxt(response, **kwargs) for response in responses]
                    self.data[item_id]["responses"] = self.data[item_id]["clean_responses"]
                    
            if ("information" in target or "all" in target) and "information" in item_data and item_data["information"]:
                info = item_data["information"]
                self.data[item_id]["clean_information"] = cleantxt(info, **kwargs)
                self.data[item_id]["information"] = self.data[item_id]["clean_information"]

    def tokenize(
        self,
        target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all"
    ) -> None:
        target = [target] if isinstance(target, str) else target
        
        for item_id, item_data in self.data.items():
            if "answer" in target or "all" in target:
                if "answers" in item_data:
                    answers = item_data["answers"] if isinstance(item_data["answers"], list) else [item_data["answers"]]
                    self.data[item_id]["token_answers"] = [token for answer in answers for token in tokenizetxt(answer)]
                    self.data[item_id]["answers"] = self.data[item_id]["token_answers"]
                    
            if "response" in target or "all" in target:
                if "responses" in item_data:
                    responses = item_data["responses"] if isinstance(item_data["responses"], list) else [item_data["responses"]]
                    self.data[item_id]["token_responses"] = [token for response in responses for token in tokenizetxt(response)]
                    self.data[item_id]["responses"] = self.data[item_id]["token_responses"]
                    
            if ("information" in target or "all" in target) and "information" in item_data and item_data["information"]:
                info = item_data["information"]
                if isinstance(info, list):
                    self.data[item_id]["token_information"] = [token for i in info for token in tokenizetxt(i)]
                else:
                    self.data[item_id]["token_information"] = tokenizetxt(info)
                self.data[item_id]["information"] = self.data[item_id]["token_information"]

    def jamoize(
        self,
        target: Union[list[Literal["answer", "response", "information", "all"]], Literal["answer", "response", "information", "all"]] = "all"
    ) -> None:
        target = [target] if isinstance(target, str) else target
        
        for item_id, item_data in self.data.items():
            if "answer" in target or "all" in target:
                if "answers" in item_data:
                    answers = item_data["answers"] if isinstance(item_data["answers"], list) else [item_data["answers"]]
                    self.data[item_id]["jamo_answers"] = [jamoizetxt(answer) for answer in answers]
                    
            if "response" in target or "all" in target:
                if "responses" in item_data:
                    responses = item_data["responses"] if isinstance(item_data["responses"], list) else [item_data["responses"]]
                    self.data[item_id]["jamo_responses"] = [jamoizetxt(response) for response in responses]
                    
            if ("information" in target or "all" in target) and "information" in item_data and item_data["information"]:
                info = item_data["information"]
                if isinstance(info, list):
                    self.data[item_id]["jamo_information"] = [jamoizetxt(i) for i in info]
                else:
                    self.data[item_id]["jamo_information"] = jamoizetxt(info)

    def format(self, information: bool = True) -> None:
        """모든 문항의 데이터를 하나의 학습용 데이터로 합침"""
        self.result = []
        
        for item_id, item_data in self.data.items():
            answers = item_data.get("answers", [])
            responses = item_data.get("responses", [])
            info = item_data.get("information") if information else None
            
            item_result = formattxt(
                answer=answers,
                response=responses,
                information=info
            )
            self.result.extend(item_result)
        
        return self.result

    def train(self, **kwargs):
        """FastText 모델 학습"""
        if not hasattr(self, 'result'):
            self.format()
        
        if self.model is None:
            self.model = FastText(sentences=self.result, **kwargs)
        else:
            self.model.build_vocab(self.result, update=True)
            self.model.train(self.result, total_examples=len(self.result), epochs=self.model.epochs)
        
        return self.model

    def fit(self, **kwargs):
        """학습 래퍼 함수"""
        return self.train(**kwargs)