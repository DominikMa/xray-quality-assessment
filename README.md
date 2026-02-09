# X-ray Quality Analyser

A tool to analyze the quality of radiographs using ML models.
If you just want a fast preview, there is a [Web Version](https://dominikma.github.io/xray-quality-assessment-web/) running entirely in the browser.

## Download the models

Download the models from [the latest release](https://github.com/DominikMa/xray-quality-assessment/releases/latest) and place all model files in a single folder.
The folder structure should match the one shown in [Models Structure](#models-structure).


## Installation

The easiest way to run the tool is using [uv](https://docs.astral.sh/uv/).
If you don't have uv installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), or alternatively use pip.

### Run it with uv
```bash
uv run --with "git+https://github.com/DominikMa/xray-quality-assessment" xray-quality-analyser <models_path> <input_folder>
```

### Install as package
If you plan to run xray-quality-analyser multiple times, installing it as a package is recommended. You can install it with your preferred package manager:

```bash
pip install "git+https://github.com/DominikMa/xray-quality-assessment"
```

Then run it with:
```bash
xray-quality-analyser <models_path> <input_folder>
```

## Usage

### CLI Command

```bash
xray-quality-analyser MODELS_PATH INPUT_FOLDER [OPTIONS]
```

**Arguments:**
- `MODELS_PATH`: Path to the folder containing the ONNX model files
- `INPUT_FOLDER`: Path to the folder containing DICOM files to analyze

**Options:**
- `-o, --output-folder PATH`: Optional output folder for predictions. If not specified, predictions are saved in the input folder.
- `--copy-file-structure / --no-copy-file-structure`:
  - `--copy-file-structure`: Creates a separate JSON file for each DICOM file (default)
  - `--no-copy-file-structure`: Creates a single JSON file containing all predictions (default: disabled)
- `--help`: Show help message and exit.

### Examples

**Basic usage:**
```bash
xray-quality-analyser ./models ./dicom_files
```

**With custom output folder:**
```bash
xray-quality-analyser ./models ./dicom_files -o ./results
```

**Single output file with predictions:**
```bash
xray-quality-analyser ./models ./dicom_files -o ./results --no-copy-file-structure
```

## Models Structure

The tool expects models to be in a flat structure with the following naming convention:

```
models/
├── body_part.proto
├── view_ankle.proto
├── view_knee.proto
├── view_wrist.proto
├── laterality_ankle_ap.proto
├── laterality_ankle_lat.proto
├── laterality_knee_ap.proto
├── laterality_knee_lat.proto
├── laterality_wrist_ap.proto
├── laterality_wrist_lat.proto
├── quality_ankle_ap.proto
├── quality_ankle_lat.proto
├── quality_knee_ap.proto
├── quality_knee_lat.proto
├── quality_wrist_ap.proto
├── quality_wrist_lat.proto
├── roi_ankle_ap.proto
├── roi_ankle_lat.proto
├── roi_knee_ap.proto
├── roi_knee_lat.proto
├── roi_wrist_ap.proto
└── roi_wrist_lat.proto
```

## Output

The tool generates JSON files with predictions. Example output:

```json
{
  "predictions": {
    "body_part": "ankle",
    "view": "ap",
    "laterality": "left",
    "quality": 1.85
  }
}
```

## Supported DICOM Files

- **Modalities**: DX (Digital Radiography), CR (Computed Radiography)
- **SOP Class UIDs**:
  - 1.2.840.10008.5.1.4.1.1.1 (CR Image Storage)
  - 1.2.840.10008.5.1.4.1.1.1.1 (Digital X-Ray Image Storage - For Presentation)
  - 1.2.840.10008.5.1.4.1.1.1.1.1 (Digital X-Ray Image Storage - For Processing)
