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

## Project Structure
```
├── app.py                   # Gradio web app (main entry point)
├── config.py                # Class labels, model paths, settings
├── train_brain_mri.py       # Training script
├── evaluate_mri_model.py    # Evaluation & confusion matrix
├── gradcam.py               # Standalone Grad-CAM visualization
├── medical_predict.py       # CLI prediction script
├── pdf_report.py            # PDF report generator
├── prepare_dataset.py       # Dataset download & organization
├── models/
│   └── brain_mri_multiclass.pth   # Trained model weights (16MB)
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

## Deployment

### Hugging Face Spaces
1. Create a new Space (Gradio SDK)
2. Upload: `app.py`, `config.py`, `pdf_report.py`, `requirements.txt`, `models/`
3. The app auto-deploys

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py config.py pdf_report.py ./
COPY models/ models/
EXPOSE 7860
CMD ["python", "app.py"]
```

### Render / Railway
1. Connect your Git repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`

## Core Files for Deployment
Only these files are needed to deploy the web app:
- `app.py`
- `config.py`
- `pdf_report.py`
- `requirements.txt`
- `models/brain_mri_multiclass.pth`

## Training
To retrain the model:
```bash
python prepare_dataset.py   # Download & organize datasets
python train_brain_mri.py   # Train EfficientNet-B0 (20 epochs)
python evaluate_mri_model.py  # Evaluate performance
```

## Tech Stack
- **Model**: EfficientNet-B0 (PyTorch)
- **UI**: Gradio
- **Reports**: ReportLab (PDF)
- **Visualization**: Grad-CAM heatmaps
