import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import models
from torchvision.transforms import v2 as transforms_v2

from model.utils import get_device

# Turkish name overrides (currently unused but kept for future use)
TURKISH_NAME_OVERRIDES: Dict[str, str] = {
    "zeytinyagli-fasulye": "Zeytinyağlı Fasulye",
    "yogurt": "Yoğurt",
    "yogurtlu-makarna": "Yoğurtlu Makarna",
    "tarhana-corbasi": "Tarhana Çorbası",
    "mercimek-corbasi": "Mercimek Çorbası",
    "domates-corbasi": "Domates Çorbası",
    "yayla-corbasi": "Yayla Çorbası",
    "sehriye-corbasi": "Şehriye Çorbası",
    "seftali": "Şeftali",
    "cig-kofte": "Çiğ Köfte",
    "icli-kofte": "İçli Köfte",
    "anne-koftesi": "Anne Köftesi",
    "mercimek-koftesi": "Mercimek Köftesi",
    "turk-kahvesi": "Türk Kahvesi",
    "iskender": "İskender",
    "ispanak-yemegi": "Ispanak Yemeği",
    "cacik": "Cacık",
    "cay": "Çay",
    "cilek": "Çilek",
    "cipura": "Çipura",
    "coban-salatasi": "Çoban Salatası",
    "doner": "Döner",
    "kokorec": "Kokoreç",
    "sutlac": "Sütlaç",
    "tursu": "Turşu",
    "uzum": "Üzüm",
    "incir": "İncir",
    "kiymali-borek": "Kıymalı Börek",
    "kiymali-pide": "Kıymalı Pide",
    "peynirli-borek": "Peynirli Börek",
    "su-boregi": "Su Böreği",
    "kabak-mucver": "Kabak Mücver",
    "hunkar-begendi": "Hünkar Beğendi",
    "kemal-pasa-tatlisi": "Kemal Paşa Tatlısı",
    "tulumba-tatlisi": "Tulumba Tatlısı",
    "canak-enginar": "Çanak Enginar",
    "bruksel-lahanasi": "Brüksel Lahanası",
    "beyaz-lahana-sarmasi": "Beyaz Lahana Sarması",
    "sulu-bamya-yemegi": "Sulu Bamya Yemeği",
    "sulu-barbunya-yemegi": "Sulu Barbunya Yemeği",
    "sulu-bezelye-yemegi": "Sulu Bezelye Yemeği",
    "sulu-kuru-fasulye-yemegi": "Sulu Kuru Fasulye Yemeği",
    "sulu-mercimek-yemegi": "Sulu Mercimek Yemeği",
    "sulu-nohut-yemegi": "Sulu Nohut Yemeği",
    "sulu-patates-yemegi": "Sulu Patates Yemeği",
    "salcali-makarna": "Salçalı Makarna",
    "haslanmis-yumurta": "Haşlanmış Yumurta",
    "sucuklu-yumurta": "Sucuklu Yumurta",
    "patlican-kebabi": "Patlıcan Kebabı",
    "tas-kebabi": "Taş Kebabı",
    "adana-kebap": "Adana Kebap",
    "et-sote": "Et Sote",
    "tavuk-sote": "Tavuk Sote",
    "midye-dolma": "Midye Dolma",
    "biber-dolma": "Biber Dolma",
    "mumbar-dolmasi": "Mumbar Dolması",
    "yaprak-sarma": "Yaprak Sarma",
    "karniyarik": "Karnıyarık",
    "patates-kizartmasi": "Patates Kızartması",
    "patates-puresi": "Patates Püresi",
    "patates-salatasi": "Patates Salatası",
    "bulgur-pilavi": "Bulgur Pilavı",
    "hamsi-tava": "Hamsi Tava",
    "midye-tava": "Midye Tava",
    "kisir": "Kısır",
    "manti": "Mantı",
    "lahmacun": "Lahmacun",
    "kalburabasti": "Kalburabastı",
    "kazandibi": "Kazandibi",
    "dondurma": "Dondurma",
    "baklava": "Baklava",
    "lokma": "Lokma",
    "sahlep": "Sahlep",
    "ayran": "Ayran",
    "menemen": "Menemen",
    "omlet": "Omlet",
    "sandvic": "Sandviç",
    "tantuni": "Tantuni",
    "pilav": "Pilav",
    "ekmek": "Ekmek",
    "levrek": "Levrek",
    "siyah-zeytin": "Siyah Zeytin",
    "yesil-zeytin": "Yeşil Zeytin",
    "domates": "Domates",
    "salatalik": "Salatalık",
    "havuc": "Havuç",
    "brokoli": "Brokoli",
    "karnabahar": "Karnabahar",
    "pirasa": "Pırasa",
    "armut": "Armut",
    "elma": "Elma",
    "erik": "Erik",
    "kayisi": "Kayısı",
    "kiraz": "Kiraz",
    "karpuz": "Karpuz",
    "kavun": "Kavun",
    "kivi": "Kivi",
    "mango": "Mango",
    "muz": "Muz",
    "nar": "Nar",
    "portakal": "Portakal",
    "avokado": "Avokado",
}


def format_class_name(raw_name: str) -> str:
    """Convert raw class name to formatted Turkish name."""
    if raw_name in TURKISH_NAME_OVERRIDES:
        return TURKISH_NAME_OVERRIDES[raw_name]
    return raw_name.replace("-", " ").replace("_", " ").title()


def create_model_from_checkpoint(
    config: Dict[str, Any], checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, List[str], float]:
    """Load model from checkpoint matching training logic."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if class_names is None:
        raise ValueError(
            "Checkpoint does not contain 'class_names'. "
            "This checkpoint may be from an older version. Please retrain."
        )

    state_dict = checkpoint["model_state_dict"]
    if "fc.weight" in state_dict:
        num_classes_from_weights = state_dict["fc.weight"].shape[0]
    elif "classifier.1.weight" in state_dict:
        num_classes_from_weights = state_dict["classifier.1.weight"].shape[0]
    else:
        num_classes_from_weights = len(class_names)

    if len(class_names) != num_classes_from_weights:
        print(
            f"Warning: Checkpoint has {num_classes_from_weights} output classes "
            f"but {len(class_names)} class_names. Trimming to match."
        )
        class_names = class_names[:num_classes_from_weights]

    if "integration_examples" in class_names:
        print(
            "Warning: 'integration_examples' found in checkpoint class_names. "
            "This is a stale entry. Please retrain the model."
        )

    num_classes = num_classes_from_weights
    backbone_name = config["model"]["backbone"]
    if backbone_name == "resnet18":
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif backbone_name == "resnet34":
        model = models.resnet34(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported: 'resnet18', 'resnet34', 'efficientnet_b0'"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"Model loaded. Validation acc: {val_acc:.2f}%")
    return model, class_names, val_acc


class TurkishFoodPredictor:
    """Predictor for Turkish food classification from image bytes."""

    def __init__(self, model_path: str, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        self.device = get_device()

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model, self.class_names, _ = create_model_from_checkpoint(
            self.config, checkpoint_path, self.device
        )

        image_size = self.config["data"]["image_size"]
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.Resize((image_size, image_size)),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        print(f"Model loaded. Classes: {len(self.class_names)}")

    def preprocess_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to preprocessed tensor."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def predict_image_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        """Predict food class from image bytes. Returns (label, confidence)."""
        image_tensor = self.preprocess_image_bytes(image_bytes)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, dim=1)

        idx = top_idx[0].item()
        prob = top_prob[0].item()
        raw_class = self.class_names[idx]

        return raw_class, float(prob)


_predictor: Optional[TurkishFoodPredictor] = None


def get_predictor() -> TurkishFoodPredictor:
    """Get global predictor instance, loading model on first call."""
    global _predictor

    if _predictor is None:
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / "model" / "best_model.pt"
        config_path = base_dir / "model" / "config.yaml"

        _predictor = TurkishFoodPredictor(str(model_path), str(config_path))

    return _predictor
