import logging
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from pydicom import FileDataset

from xray_quality_analyser.transforms import (
    get_roi_for_theta,
    prepare_input,
    prepare_input_roi,
)


class Model(ABC):
    PREFIX = None
    OUTPUT_NAMES = None

    def __init__(self, model_path: Path, name: str) -> None:
        self.model_path = model_path
        self.name = name

        self.onnx_session = onnxruntime.InferenceSession(model_path)

    def _predict_raw(self, inputs: Dict[str, np.ndarray], output_names: List[str] | None = None):
        return self.onnx_session.run(output_names, inputs)

    def predict_raw(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        model_output = np.array(self._predict_raw(inputs, output_names=self.OUTPUT_NAMES)).squeeze()
        return model_output

    def predict(self, inputs: Dict[str, np.ndarray]):
        raise NotImplementedError


class RegressionModel(Model, ABC):
    def predict(self, inputs: Dict[str, np.ndarray]) -> float:
        return float(self.predict_raw(inputs))


class ClassificationModel(Model, ABC):
    CLASS_ID_TO_NAME: Dict[int, str] = {}

    def predict(self, inputs: Dict[str, np.ndarray]) -> str:
        logits = self.predict_raw(inputs)
        class_id = int(logits.argmax())
        return self.CLASS_ID_TO_NAME[class_id]


class BodyPartModel(ClassificationModel):
    PREFIX = "body_part"
    OUTPUT_NAMES = ["body_part"]
    CLASS_ID_TO_NAME = {
        0: "ankle",
        1: "knee",
        2: "wrist",
    }


class LateralityModel(ClassificationModel):
    PREFIX = "laterality"
    OUTPUT_NAMES = ["laterality"]
    CLASS_ID_TO_NAME = {
        0: "left",
        1: "right",
    }


class QualityModel(RegressionModel):
    PREFIX = "quality"
    OUTPUT_NAMES = ["quality"]


class ROIModel(Model):
    PREFIX = "roi"
    OUTPUT_NAMES = ["roi_theta"]

    def predict(self, inputs: Dict[str, np.ndarray]):
        return self.predict_raw(inputs)


class ViewModel(ClassificationModel):
    PREFIX = "view"
    OUTPUT_NAMES = ["view"]
    CLASS_ID_TO_NAME = {
        0: "ap",
        1: "lat",
    }


class StandalonePredictorBase:
    SUPPORTED_MODALITIES = {"DX", "CR"}
    SUPPORTED_SOPS = {
        "1.2.840.10008.5.1.4.1.1.1",
        "1.2.840.10008.5.1.4.1.1.1.1",
        "1.2.840.10008.5.1.4.1.1.1.1.1",
    }

    def __init__(self, models_path: Path) -> None:
        super().__init__()
        self.logger = logging.getLogger("Predictor")
        self.models_path = models_path
        self.models = self.load_models()

    def load_models(self):
        models_base_path = self.models_path
        models = defaultdict(dict)

        self.logger.info("Loading models from: %s", models_base_path)
        for model_file in models_base_path.glob("*.proto"):
            filename = model_file.stem
            parts = filename.split("_")

            if parts[0] == "body" and parts[1] == "part":
                models["BodyPartModel"][()] = BodyPartModel(model_path=model_file, name="body-part")
            elif parts[0] == "view":
                body_part = parts[1]
                models["ViewModel"][(body_part,)] = ViewModel(model_path=model_file, name=f"view-{body_part}")
            elif parts[0] == "laterality":
                body_part = parts[1]
                view = parts[2]
                models["LateralityModel"][(body_part, view)] = LateralityModel(
                    model_path=model_file, name=f"laterality-{body_part}-{view}"
                )
            elif parts[0] == "quality":
                body_part = parts[1]
                view = parts[2]
                models["QualityModel"][(body_part, view)] = QualityModel(
                    model_path=model_file, name=f"quality-{body_part}-{view}"
                )
            elif parts[0] == "roi":
                body_part = parts[1]
                view = parts[2]
                models["ROIModel"][(body_part, view)] = ROIModel(model_path=model_file, name=f"roi-{body_part}-{view}")
        self.logger.info("loaded models")
        return models

    def predict_body_part(self, input_image: np.ndarray) -> str:
        self.logger.debug("predicting body part")
        model: BodyPartModel = self.models["BodyPartModel"][()]
        return model.predict({"xray_image": input_image})

    def predict_view(self, input_image: np.ndarray, body_part: str) -> str:
        self.logger.debug("predicting view")
        model: ViewModel = self.models["ViewModel"][(body_part,)]
        return model.predict({"xray_image": input_image})

    def predict_laterality(self, input_image: np.ndarray, body_part: str, view: str) -> str:
        self.logger.debug("predicting body laterality")
        model: LateralityModel = self.models["LateralityModel"][
            (
                body_part,
                view,
            )
        ]
        return model.predict({"xray_image": input_image})

    def predict_roi(self, input_image: np.ndarray, body_part: str, view: str) -> np.ndarray:
        self.logger.debug("predicting roi")
        model: ROIModel = self.models["ROIModel"][
            (
                body_part,
                view,
            )
        ]
        theta = model.predict({"xray_image": prepare_input_roi(input_image)})
        roi = get_roi_for_theta(input_image, theta)
        return roi

    def predict_quality(self, input_image: np.ndarray, body_part: str, view: str) -> float:
        self.logger.debug("predicting quality")
        model: QualityModel = self.models["QualityModel"][
            (
                body_part,
                view,
            )
        ]
        return model.predict({"xray_image_roi": input_image})

    def predict(self, input_image: np.ndarray) -> Tuple[str, str, str, float, np.ndarray]:
        prepared_input = prepare_input(input_image, 224, 4)
        body_part = self.predict_body_part(prepared_input)
        view = self.predict_view(prepared_input, body_part)
        laterality = self.predict_laterality(prepared_input, body_part, view)

        roi = self.predict_roi(input_image, body_part, view)
        quality = self.predict_quality(prepare_input(roi.copy(), 224, 4), body_part, view)

        return body_part, view, laterality, quality, roi

    def should_skip(self, dataset: FileDataset, logger: Optional[logging.Logger] = None) -> bool:
        if not hasattr(dataset, "Modality") or str(dataset.Modality).upper() not in self.SUPPORTED_MODALITIES:
            if logger:
                logger.info("skip dicom because modality %s s not supported", getattr(dataset, "Modality", "N/A"))
            return True

        if not hasattr(dataset, "SOPClassUID") or str(dataset.SOPClassUID) not in self.SUPPORTED_SOPS:
            if logger:
                logger.info(
                    "skip dicom because sop class uid %s is not supported", getattr(dataset, "SOPClassUID", "N/A")
                )
            return True

        return False

    def run(self):
        raise NotImplementedError
