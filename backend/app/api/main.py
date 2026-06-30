import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.data.taxonomy import TaxonomyDB
from app.inference.engine import InferenceEngine
from app.inference.formatter import ResultFormatter
from app.models.gradcam import GradCAMEngine

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="BiT Bacterial Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference = InferenceEngine(checkpoint_path=os.getenv("MODEL_CHECKPOINT"))
gradcam = GradCAMEngine()
formatter = ResultFormatter()
taxonomy = TaxonomyDB()


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model": "loaded" if inference.model_loaded else "demo"}


@app.get("/species")
def species() -> list[dict]:
    return taxonomy.get_all()


@app.get("/species/{species_id}")
def species_by_id(species_id: int) -> dict:
    record = taxonomy.query_species(species_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Species not found")
    return record


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10MB limit")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Only image uploads are supported")
    try:
        prediction = inference.predict_bytes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cam = gradcam.compute_cam(prediction["image"], prediction["predicted_class_id"])
    heatmap = gradcam.overlay_base64(prediction["image"], cam)
    return formatter.format(prediction, heatmap)
