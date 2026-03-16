---
title: Brain MRI Diagnostic System
emoji: 🧠
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: AI Brain MRI classifier with 99.87% accuracy
---

# Brain MRI AI Diagnostic System

Multi-class brain MRI classifier using **EfficientNet-B0** with **99.87% accuracy**.

## Classes
| # | Class | Description |
|---|-------|-------------|
| 1 | Normal | Healthy brain scan |
| 2 | Brain Tumor | Abnormal mass lesion |
| 3 | Stroke | Ischemic territory signal change |
| 4 | Hemorrhage | Intracranial hemorrhage |
| 5 | Multiple Sclerosis | Periventricular white-matter lesions |
| 6 | Alzheimer's | Diffuse cortical atrophy |

## Features
- **6-class AI classification** with EfficientNet-B0
- **Grad-CAM heatmaps** showing model attention regions
- **Professional PDF reports** with diagnosis, differentials, scan images, probability bars
- **2-page UI**: Patient input → Analysis results
- **Stitch AI frontend** with Tailwind CSS + Material Symbols

## Project Structure
```
├── app.py                       # FastAPI backend (main entry point)
├── config.py                    # Class labels, model paths, settings
├── templates/
│   └── index.html               # Stitch AI frontend (Tailwind + Material Symbols)
├── models/
│   └── brain_mri_multiclass.pth # Trained model weights (16MB)
├── train_brain_mri.py           # Training script
├── evaluate_mri_model.py        # Evaluation & confusion matrix
├── gradcam.py                   # Standalone Grad-CAM visualization
├── prepare_dataset.py           # Dataset download & organization
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```
Opens at `http://localhost:7860`

## Tech Stack
- **Model**: EfficientNet-B0 (PyTorch)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML + Tailwind CSS + Material Symbols (from Stitch AI)
- **Reports**: ReportLab (PDF)
- **Visualization**: Grad-CAM heatmaps

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py config.py ./
COPY templates/ templates/
COPY models/ models/
EXPOSE 7860
CMD ["python", "app.py"]
```

### Render / Railway
1. Connect your Git repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`

### Core Files for Deployment
Only these files are needed to deploy the web app:
- `app.py` — FastAPI server + ML inference + PDF generation
- `config.py` — Class labels, model paths, settings
- `templates/index.html` — Frontend UI
- `models/brain_mri_multiclass.pth` — Trained model weights
- `requirements.txt` — Python dependencies

## Training
To retrain the model:
```bash
python prepare_dataset.py      # Download & organize datasets
python train_brain_mri.py      # Train EfficientNet-B0 (20 epochs)
python evaluate_mri_model.py   # Evaluate performance
```
