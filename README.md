# SpatialUncertain

---
### *Seeing Isn’t Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?*

[![arXiv](https://img.shields.io/badge/arXiv-2605.30557-b31b1b.svg)](https://arxiv.org/abs/2605.30557)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zhangyuejoslin.github.io/spatialuncertain/)

**Authors:** Yue Zhang, Zun Wang, Han Lin, Yonatan Bitton, Idan Szpektor, Mohit Bansal

## 📌 Overview

![Teaser](https://raw.githubusercontent.com/zhangyuejoslin/spatialuncertain/main/images/teaser_img.png)

---

## Setup

### 1. Install dependencies

Follow the instructions at [Holodeck](https://github.com/allenai/Holodeck) to install AI2THOR and generate 3D indoor scenes. Our generated scenes are available on [Google Drive](https://drive.google.com/drive/folders/1RDGHQd-pqHZtNzvIZQSYSRgVZwm2gmw3?usp=sharing).

```bash
pip install compress_json trimesh Pillow imageio numpy google-generativeai openai
```

### 2. Set API keys

```bash
export AZURE_OPENAI_API_KEY="your-azure-key"  # For GPT / Azure OpenAI
export GEMINI_API_KEY="your-gemini-key"        # For Gemini
```

---

## Usage

### Step 1 — Generate scene layouts

```bash
python gen_occlusion_scene.py --batch-root /path/to/scenes/       # Occlusion scenes
python gen_distortion_scene.py --input /path/to/scene_folder      # perspective ambiguity scenes
```

### Step 2 — Render scenes

```bash
python render_occlusion.py --scene-dir /path/to/occlusion_scene/ --port 
python render_distortion.py --scene-dir /path/to/distortion_scene/ --port 
```

> Requires a running AI2THOR Unity build. Set `--asset-dir` if assets are not at the default path.

### Step 3 — Generate questions

```bash
python generate_questions/question_gen_occlusion.py --batch-root /path/to/occlusion_scenes/
python generate_questions/question_gen_distortion.py --batch-root /path/to/distortion_scenes/
```

### Step 4 — Evaluate models

```bash
python eval_gpt5.py \
    --benchmark questions/occlusion_questions.json \
    --model gpt-4o \
    --provider azure \
    --output-json results/gpt4o_occlusion.json

python eval_gemini.py \
    --benchmark questions/occlusion_questions.json \
    --model gemini-2.5-flash \
    --output-json results/gemini25flash_occlusion.json
```

---

## Dataset

Our benchmark dataset is available on [HuggingFace](https://huggingface.co/datasets/Yuezhangjoslin/spatialuncertain).

---

## Notes

- All scripts expect AI2THOR assets to be present at `OBJATHOR_ASSETS_DIR` (set in `ai2holodeck/constants.py`).
- Rendered images and scene metadata are saved alongside the scene folders by default.
- Evaluation results are saved as JSON; a `.jsonl` file is also saved to track per-question predictions.

---

## Citation

```bibtex
@article{zhang2025spatialuncertain,
  title     = {Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?},
  author    = {Zhang, Yue and Wang, Zun and Lin, Han and Bitton, Yonatan and Szpektor, Idan and Bansal, Mohit},
  journal   = {arXiv preprint arXiv:2605.30557},
  year      = {2026}
}
```
