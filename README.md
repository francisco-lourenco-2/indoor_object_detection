# DocuSketch — Indoor Object Detection

Solution for the DocuSketch AI/ML Engineer home assignment: train and evaluate an object detector on the [TUT Indoor Object Detection Dataset](https://zenodo.org/record/2654485).

The main deliverable is the notebook **`docusketch_inddor_object_detection.ipynb`**, which walks through the full pipeline end to end. The Python modules in this folder are the implementation behind that notebook.

## What the notebook does

1. **Setup** — install dependencies, set paths  
2. **Download dataset** — Zenodo indoor detection data  
3. **Dataset inspection** — stats, interactive bbox viewer (`browse_dataset_notebook`)  
4. **Stratified split** — 80/10/10 train/val/test via **ILP** (`generate_splits.py`) so class balance is preserved (important for rare classes like printer/screen)  
5. **YOLO preparation** — XML → Ultralytics format (`prepare_yolo_dataset.py`)  
6. **Training & evaluation** — **YOLOv8n** fine-tuning (`train_and_eval.py`); demo cell loads best weights from **`experiment_2`**  
7. **Visualization** — validation mAP table + best/worst prediction montages (`visualize_predictions.py`)

## Main result (local run `experiment_2`)

| Metric (test) | Value |
|---------------|------:|
| mAP50-95 | 0.851 |
| mAP50 | 0.981 |
| mAP75 | 0.960 |

Trained 100 epochs with default Ultralytics augmentations. A no-augmentation baseline reached ~0.74 mAP50-95.

## Project layout

| Path | Role |
|------|------|
| `docusketch_inddor_object_detection.ipynb` | Report + runnable pipeline |
| `analyse_data.py` | Download, stats, dataset browser |
| `generate_splits.py` | ILP-based stratified splits → `train.txt` / `valid.txt` / `test.txt` |
| `prepare_yolo_dataset.py` | YOLO dataset + `indoor.yaml` |
| `train_and_eval.py` | YOLOv8 training and metrics |
| `visualize_predictions.py` | Qualitative val/test panels |
| `exp_logging/` | Experiment tracking under `work_dirs/` (metrics, checkpoints, logs) |
| `requirements.txt` | Python dependencies |

## Running locally

```bash
cd docusketch_assignment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook docusketch_inddor_object_detection.ipynb
```

Use a **GPU** for training. Notebook training uses `build_reports=False` (no OpenAI). Optional reports locally: `export OPENAI_API_KEY=...` and `python train_and_eval.py --reports` (interactive CLI).

## Google Colab

1. **Runtime → Change runtime type → GPU**
2. Run the **Colab bootstrap** cell (clone project), then **Setup**, then the rest top to bottom.

**Clone error** (`could not read Username for 'https://github.com'`): the repo is **private**. Use one of:

- **Make the repo public** (easiest for reviewers), then:
  `git clone https://github.com/francisco-lourenco-2/dokusketch_indoor_object_detection_assignment.git /content/docusketch_assignment`
- **Private repo:** Colab → **Secrets** (key icon) → add `GITHUB_TOKEN` (GitHub PAT with `repo` scope) → use **Option B** in the bootstrap cell.
- **Google Drive:** zip the project, upload, unzip with **Option C** in the bootstrap cell.

Repo URL: `https://github.com/francisco-lourenco-2/dokusketch_indoor_object_detection_assignment`

The dataset is downloaded in-notebook. Ship `work_dirs/.../experiment_2/ckpts/best.pt` via Release or Drive for the demo training cell.

## Not in git (see `.gitignore`)

- `data/` — raw dataset (downloaded in notebook)  
- `prepared_data/` — generated YOLO export  
- `work_dirs/` — experiment outputs and checkpoints  
- `*.pt` — weight files  

Pre-generated results for the write-up live under `work_dirs/indoor_object_detection/original/yolov8n/experiment_2/` when present locally.
