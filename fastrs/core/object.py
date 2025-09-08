import jamo
from gensim.models import FastText
from typing import Union, Literal, Dict, Any, List
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from . import util
from .preprocessor import Preprocessor
from .visualizer import visualize_embeddings


class Fastrs:

    @overload
    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]],
        



    @util.copysignature(FastText.__init__)
    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, list[str], None]]] = None,
        answers: np.ndarray = None,
        responses: np.ndarray = None,
        informations: np.ndarray = None,
        model: Union[FastText, None] = None,
        option: Literal["automatic", "manual"] = "automatic",
        fasttext_preset: Literal["default", "fast_training", "high_quality"] = "default",
        **params: Dict[str, Any],
    ) -> None:
        """
        FastRS 초기화
        
        Args:
            data: 딕셔너리 형태의 데이터
            answers: 정답 배열
            responses: 응답 배열  
            informations: 정보 배열
            model: 기존 FastText 모델
            option: 처리 옵션 ("automatic", "manual")
            fasttext_preset: FastText 설정 프리셋 ("default", "fast_training", "high_quality")
            **params: 추가 FastText 파라미터 (프리셋을 덮어씀)
        """

        # FastText 설정 로드 및 적용
        fasttext_defaults = util.load_fasttext_defaults()
        self.fasttext_params = fasttext_defaults[fasttext_preset].copy()
        self.fasttext_params.update(params)  # 사용자 파라미터로 덮어쓰기
        
        # epochs를 별도 속성으로 저장 (finetune에서 사용)
        self.epochs = self.fasttext_params.pop('epochs', 5)

        #set Attr
        self.model = model
        if data is not None : self.data = data
        else: self.data = util.formatData(answers, responses, informations)
        util.validData(self.data)

        if option == "automatic":
            traindata = []
            for item in self.data.values():
                preprocessed = Preprocessor(
                    answers=item["answer"],
                    responses=item["response"],
                    information=item.get("information", None),
                    option="automatic"
                )
                traindata.append(preprocessed)
            self.traindata = traindata
            self.sentences = [sent for pp in self.traindata for sent in pp]
            self.finetune(model if model is not None else util.get_pretrained_model())
        elif option == "manual": pass

    def finetune(
            self,
            model: FastText = None
    ) -> FastText:
        """
        기존 FastText 모델 파인튜닝
        
        Args:
            model: 파인튜닝할 기존 모델 (없으면 사전훈련된 모델 사용)
            
        Returns:
            FastText: 파인튜닝된 모델
        """
        model = model or self.model or util.get_pretrained_model()
        model.build_vocab(self.sentences, update=True, trim_rule=None)
        model.train(
            corpus_iterable=self.sentences,
            total_examples=len(self.sentences),
            epochs=self.epochs,
        )
        self.model = model
        return model

    def train(
            self,
    ) -> FastText:
        """
        새로운 FastText 모델 훈련
        
        Returns:
            FastText: 훈련된 모델
        """
        model = FastText(
            sentences=self.sentences,
            **self.fasttext_params
        )
        self.model = model
        return model
    
    def reduce(
        self,
        method : Literal["umap", "pca", "tsne"] = "umap",
        **method_params: Dict[str, Any],
    ) -> pd.DataFrame:
        if method == "umap":
            reducer = UMAP(n_components=2, **method_params)
        elif method == "pca":
            reducer = PCA(n_components=2, **method_params)
        elif method == "tsne":
            reducer = TSNE(n_components=2, **method_params)
        else:
            raise ValueError(f"Unknown method: {method}")
        reduced_vectors = reducer.fit_transform(self.model.wv.vectors)
        tokens = list(self.model.wv.key_to_index.keys())
        demension2_data = pd.DataFrame(reduced_vectors, columns=["x", "y"])
        token_data = pd.DataFrame(tokens, columns=["token"])
        result = pd.concat([demension2_data, token_data], axis=1)
        result = result[["token", "x", "y"]]
        return result

    def _create_embeddings_dataframe(self, reduced_data: pd.DataFrame) -> pd.DataFrame:
        """임베딩을 DataFrame으로 변환 (차원축소 데이터 활용)"""
        kv = self.model.wv
        words = list(kv.key_to_index.keys())
        
        # reduced_data에 token 정보 추가
        df = reduced_data.copy()
        df["token"] = words[:len(df)]  # 길이가 맞지 않을 수 있으므로 조정
        
        return df

    def _add_answer_info(
        self, 
        df: pd.DataFrame, 
        answers: Union[str, List[str]], 
        conversion_dict: Dict = None,
        label_dict: Dict = None
    ) -> pd.DataFrame:
        """DataFrame에 정답 정보 추가"""
        if isinstance(answers, str):
            answers = [answers]
        
        # 자모 변환
        answer_jamo_list = [jamo.j2hcj(jamo.h2j(ans)) for ans in answers]
        
        # 정답 컬럼 초기화
        df["answer"] = None
        df["label"] = "0"  # 기본값을 "0"으로 설정
        df["answer_vis"] = ""
        df["token_vis"] = '"' + df["token"] + '"'
        
        # 정답 토큰들에 대해 정보 설정
        for ans_jamo, original_ans in zip(answer_jamo_list, answers):
            if ans_jamo in df["token"].values:
                idx = df[df["token"] == ans_jamo].index[0]
                df.loc[idx, "answer"] = original_ans
                df.loc[idx, "label"] = "1"  # 정답은 레이블 "1"
        
        # conversion_dict와 label_dict가 있으면 추가 정보 설정
        if conversion_dict and label_dict:
            for idx, token in enumerate(df["token"]):
                original = conversion_dict.get(token)
                if original:
                    label = label_dict.get(original)
                    df.loc[idx, "label"] = str(label) if label is not None else "0"
                    df.loc[idx, "answer_vis"] = f'"{original}"'
        
        return df

    def visualize(
        self,
        answers: Union[str, List[str]] = None,
        method: Literal["umap", "pca", "tsne"] = "umap",
        conversion_dict: Dict = None,
        label_dict: Dict = None,
        color_scheme: Literal["default", "colorblind_friendly"] = "default",
        title: str = None,
        show: bool = True,
        **method_params
    ) -> Any:
        """
        임베딩 시각화
        
        Args:
            answers: 시각화할 정답 (없으면 자동으로 데이터에서 추출)
            method: 차원축소 방법 ("umap", "pca", "tsne")
            conversion_dict: 토큰-원본 변환 딕셔너리
            label_dict: 원본-레이벨 딕셔너리
            color_scheme: 색상 스키마 ("default", "colorblind_friendly")
            title: 그래프 제목
            show: 그래프 표시 여부
            **method_params: 차원축소 알고리즘 파라미터
        
        Returns:
            plotly Figure 객체
        """
        if self.model is None:
            raise ValueError("모델이 없습니다. train() 또는 finetune()을 먼저 실행하세요.")
        
        # answers가 없으면 데이터에서 추출
        if answers is None:
            all_answers = []
            for item_data in self.data.values():
                if isinstance(item_data["answer"], list):
                    all_answers.extend(item_data["answer"])
                else:
                    all_answers.append(item_data["answer"])
            answers = list(set(all_answers))[:5]  # 최대 5개까지
        
        # 1. 기존 reduce() 메서드를 활용하여 차원축소
        reduced_data = self.reduce(method=method, **method_params)
        
        # 2. DataFrame 생성 (x, y, token 컬럼)
        df = self._create_embeddings_dataframe(reduced_data)
        
        # 3. 정답 정보 추가 (label, answer_vis, token_vis, answer 컬럼)
        df = self._add_answer_info(df, answers, conversion_dict, label_dict)
        
        # 4. 순수한 visualizer 함수 호출
        return visualize_embeddings(
            df=df,
            answers=answers,
            color_scheme=color_scheme,
            title=title,
            show=show
        )
