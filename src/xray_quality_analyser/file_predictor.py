import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from threading import Thread

import click
import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

from xray_quality_analyser.base import StandalonePredictorBase
from xray_quality_analyser.utils import dicom_to_png, setup_logger


class FilePredictor(StandalonePredictorBase):
    def __init__(
        self,
        input_folder_path: Path,
        models_path: Path,
        output_folder_path: Path | None,
        copy_file_structure: bool = True,
    ):
        super().__init__(models_path=models_path)
        self.log_dir = output_folder_path if output_folder_path else input_folder_path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir.joinpath("predictor.log")
        setup_logger(
            "Predictor",
            file_log_path=self.log_file_path,
            file_log_level=logging.INFO,
            console_log_level=logging.WARNING,
        )

        self.input_folder_path = input_folder_path
        self.copy_file_structure = copy_file_structure

        self.logger.info("run prediction")
        self.run()

    def run(self):
        file_paths = list(filter(Path.is_file, self.input_folder_path.rglob("*")))
        postfix = defaultdict(lambda: 0)

        single_file_data = {} if not self.copy_file_structure else None

        t = tqdm(smoothing=0.1, total=len(file_paths))
        for file_number, file_path in enumerate(file_paths):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dataset = dcmread(file_path, stop_before_pixels=True)
            except InvalidDicomError:
                postfix["no-dicom"] += 1
                t.total -= 1
                t.set_postfix(postfix)
                continue
            except OSError as e:
                self.logger.error("OS error reading file %s: %s", file_path, e)
                postfix["error"] += 1
                t.set_postfix(postfix)
                continue
            t.update(1)

            if self.should_skip(dataset, self.logger):
                postfix["skipped"] += 1
                t.set_postfix(postfix)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    dataset = dcmread(file_path, stop_before_pixels=False)
                except OSError as e:
                    self.logger.error("OS error reading file %s: %s", file_path, e)
                    postfix["error"] += 1
                    t.set_postfix(postfix)
                    continue

            try:
                pixel_array: np.ndarray = dicom_to_png(dataset)
                body_part, view, laterality, quality, roi = self.predict(pixel_array)
            except (ValueError, AssertionError):
                postfix["invalid-dicom"] += 1
                t.set_postfix(postfix)
                continue

            postfix[body_part] += 1
            t.set_postfix(postfix)

            data = {
                "predictions": {
                    "body_part": body_part,
                    "view": view,
                    "laterality": laterality,
                    "quality": quality,
                }
            }

            relative_path = file_path.relative_to(self.input_folder_path)
            if self.copy_file_structure:
                output_path = self.log_dir.joinpath(relative_path.with_suffix(".json"))
                Thread(
                    target=self.save_json_data,
                    args=[data, output_path],
                ).start()
            elif single_file_data is not None:
                output_path = self.log_dir.joinpath("predictions.json")
                single_file_data[str(relative_path)] = data
                if file_number % 100 == 0:
                    Thread(
                        target=self.save_json_data,
                        args=[single_file_data.copy(), output_path],
                    ).start()

    def save_json_data(self, data: dict, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@click.command()
@click.argument("models_path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--output-folder",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output folder for predictions. Defaults to input folder if not specified.",
)
@click.option(
    "--copy-file-structure/--no-copy-file-structure",
    default=False,
    help="Whether to copy the file structure from input to output folder.",
)
def main(
    models_path: Path,
    input_folder: Path,
    output_folder: Path | None,
    copy_file_structure: bool,
):
    """Run X-ray quality predictions on DICOM files in INPUT_FOLDER.

    This tool processes DICOM files and generates predictions for body part,
    view, laterality, and quality scores.
    """
    FilePredictor(
        input_folder_path=input_folder,
        output_folder_path=output_folder,
        copy_file_structure=copy_file_structure,
        models_path=models_path,
    )


if __name__ == "__main__":
    main()
