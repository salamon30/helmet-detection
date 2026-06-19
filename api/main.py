from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from api.inference import get_model, predict

app = FastAPI(
    title="Helmet Detection API",
    description="YOLOv8s PPE compliance detection — detects helmets on construction sites.",
    version="1.0.0",
)


@app.on_event("startup")
def load_model_on_startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "helmet_v3_best.pt"}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Image file (JPG/PNG)"),
    conf: float = Query(0.55, ge=0.1, le=0.95, description="Confidence threshold"),
) -> JSONResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/bmp", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image format")

    image_bytes = await file.read()
    try:
        result = predict(image_bytes, conf_threshold=conf)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return JSONResponse(content=result)
