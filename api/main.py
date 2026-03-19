from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import fastrs
from fastrs.core.util import get_pretrained_model


_pretrained_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pretrained_model
    _pretrained_model = get_pretrained_model()
    yield


app = FastAPI(
    title="SimARS API",
    description="FastText-based Response Similarity Analyzer for educational assessment",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemData(BaseModel):
    information: str
    answer: List[str]
    response: List[str]


class AnalyzeRequest(BaseModel):
    data: Dict[str, ItemData]
    reduction_method: Optional[str] = "umap"
    finetune_epochs: Optional[int] = 5
    scatter_type: Optional[str] = "simple"


class AnalyzeResponse(BaseModel):
    figures: Dict[str, str]  # item_name -> plotly figure JSON


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pretrained_model is not None}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _pretrained_model is None:
        raise HTTPException(status_code=503, detail="Pretrained model not loaded yet")

    data = {k: v.model_dump() for k, v in req.data.items()}

    try:
        analyzer = fastrs.Fastrs(data=data)
        analyzer.preprocess()
        analyzer.finetune(model=_pretrained_model, epochs=req.finetune_epochs)
        analyzer.reduce(method=req.reduction_method)
        figs = analyzer.visualize()
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = {}
    for item, fig in zip(analyzer.items, figs):
        result[item.name] = fig.to_json()

    return AnalyzeResponse(figures=result)


@app.post("/api/score")
def score(req: AnalyzeRequest):
    """
    Analyze and return cosine similarity scores of each response vs. answer.
    """
    if _pretrained_model is None:
        raise HTTPException(status_code=503, detail="Pretrained model not loaded yet")

    data = {k: v.model_dump() for k, v in req.data.items()}

    try:
        analyzer = fastrs.Fastrs(data=data)
        analyzer.preprocess()
        analyzer.finetune(model=_pretrained_model, epochs=req.finetune_epochs)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    scores = {}
    model = analyzer.model
    for item in analyzer.items:
        item_scores = {}
        answer_vecs = [
            model.wv[tok]
            for tok in item.answer
            if tok in model.wv
        ]
        if not answer_vecs:
            scores[item.name] = {}
            continue
        import numpy as np
        answer_vec = np.mean(answer_vecs, axis=0)
        for resp, orig_resp in zip(item.response, item.original_response):
            resp_tokens = resp if isinstance(resp, list) else [resp]
            resp_vecs = [model.wv[tok] for tok in resp_tokens if tok in model.wv]
            if not resp_vecs:
                item_scores[orig_resp] = None
                continue
            resp_vec = np.mean(resp_vecs, axis=0)
            sim = float(
                np.dot(answer_vec, resp_vec)
                / (np.linalg.norm(answer_vec) * np.linalg.norm(resp_vec) + 1e-8)
            )
            item_scores[orig_resp] = round(sim, 4)
        scores[item.name] = item_scores

    return {"scores": scores}
