# VISIOSCAN AI - MVP Inference Engine

## Directory Structure
```
VISIOSCAN_MVP/
├── README.md
├── requirements.txt
├── setup.py
├── inference_engine/
│   ├── inference.py
│   ├── models/
│   │   ├── cnn_ensemble.py
│   │   ├── unet3d.py
│   │   └── utils.py
│   ├── kernels/
│   │   ├── CMakeLists.txt
│   │   ├── cuda_kernel.cu
│   │   └── cuda_module.cpp
│   └── services/
│       └── api.py
```

--- README.md ---
# VISIOSCAN AI - MVP

This project provides an AI Inference Engine for real-time pathology detection on Chest X-Rays (CXR) and Head CT scans using an ensemble of 2D CNNs and a 3D U-Net, accelerated by custom CUDA kernels via PyBind11.

## Features
- **CXR detection:** Ensemble of CNNs for binary/multi-label classification.
- **Head CT segmentation:** 3D U-Net for masking intracranial hemorrhages.
- **CUDA acceleration:** Custom kernels for preprocessing/postprocessing.
- **REST API:** FastAPI service for real-time inference.

## Requirements
- Python 3.11
- PyTorch 2.2
- CUDA Toolkit (11.x or compatible)
- CMake >= 3.18
- PyBind11

## Installation
```bash
# Clone repo
git clone https://github.com/your-org/VISIOSCAN_MVP.git
cd VISIOSCAN_MVP

# Install Python deps
pip install -r requirements.txt

# Build CUDA kernels
mkdir -p inference_engine/kernels/build && cd inference_engine/kernels/build
cmake .. && make
cd ../../..

# (Optional) Install package
pip install -e .
```

## Running the API
```bash
uvicorn inference_engine.services.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `POST /infer/cxr` : Upload a single CXR image (PNG/JPEG) → returns pathology scores
- `POST /infer/head_ct` : Upload a series of DICOM files (.dcm) as a .zip → returns 3D segmentation mask

--- requirements.txt ---
fastapi
uvicorn
torch==2.2.0
numpy
pydantic
pybind11>=2.10
pillow
pydicom
scikit-image
