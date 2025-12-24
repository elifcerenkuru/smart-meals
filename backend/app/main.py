from typing import Dict, Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.foods_db import calculate_nutrition, get_food
from app.predictor import get_predictor

app = FastAPI(
    title="Tez Food Classifier API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Tez food classifier API is running"}


@app.get("/foods/{label}")
async def get_food_info(label: str) -> Dict[str, Any]:
    """Get food metadata by label."""
    food = get_food(label)
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")

    return {
        "label": label,
        **food,
    }


@app.get("/foods/{label}/nutrition")
async def get_food_nutrition(
    label: str,
    portion_count: float = 1.0,
    grams: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate nutrition based on portion count or grams.
    If grams is provided, uses per100g data; otherwise uses perPortion * portion_count.
    """
    nutrition = calculate_nutrition(label, portion_count=portion_count, grams=grams)
    if nutrition is None:
        raise HTTPException(status_code=404, detail="Food or nutrition data not found")

    food = get_food(label)
    return {
        "label": label,
        "displayName": food.get("displayName") if food else label,
        "defaultPortionGrams": food.get("defaultPortionGrams") if food else None,
        "portionCount": portion_count,
        "grams": grams,
        "nutrition": nutrition,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict food class from uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        predictor = get_predictor()
        label, confidence = predictor.predict_image_bytes(image_bytes)

        return {
            "label": label,
            "confidence": confidence,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )
