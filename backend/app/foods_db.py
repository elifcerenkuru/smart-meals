from pathlib import Path
from typing import Any, Dict, Optional

import json

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "foods.json"

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        FOODS: Dict[str, Any] = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Foods data file not found: {DATA_PATH}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in foods data file: {e}")


def get_food(label: str) -> Optional[Dict[str, Any]]:
    """Get food metadata by label. Returns None if not found."""
    return FOODS.get(label)


def calculate_nutrition(
    label: str,
    portion_count: float = 1.0,
    grams: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Calculate nutrition values.
    If grams is provided, uses per100g data; otherwise uses perPortion * portion_count.
    """
    food = get_food(label)
    if not food:
        return None

    if grams is not None:
        per100g = food.get("per100g")
        if not per100g:
            return None
        factor = grams / 100.0
        return {
            "calories": per100g["calories"] * factor,
            "carb": per100g["carb"] * factor,
            "protein": per100g["protein"] * factor,
            "fat": per100g["fat"] * factor,
        }

    per_portion = food.get("perPortion")
    if not per_portion:
        return None

    return {
        "calories": per_portion["calories"] * portion_count,
        "carb": per_portion["carb"] * portion_count,
        "protein": per_portion["protein"] * portion_count,
        "fat": per_portion["fat"] * portion_count,
    }
