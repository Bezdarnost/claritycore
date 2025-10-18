# Datasets

This project ships with **ready-to-use mirrors on Hugging Face** for fast, convenient access from scripts, notebooks, and training jobs.

If you run into any issues or want us to mirror an additional dataset, please open a GitHub issue.

---

## 📦 Available Mirrors

| Dataset            | Mirror                                                                               | Notes                                                              |
| ------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| **DF2K (bicubic)** | [`bezdarnost/DF2K-bicubic`](https://huggingface.co/datasets/bezdarnost/DF2K-bicubic) | HR images + bicubic-downsampled LR. Commonly used for SR ×2/×3/×4. |

> Tip: Mirrors are organized as plain folders so you can use them with any “image folder” loader without custom code.

---

## ⚡ Quick Start

### Option A — Stream/Load directly with 🤗 `datasets`

```python
from datasets import load_dataset

# As a generic imagefolder
ds = load_dataset(
    "imagefolder",
    data_dir="hf://datasets/bezdarnost/DF2K-bicubic",
    # split="train",
    # If you need local files instead of streaming:
    # streaming=False
)

sample = ds[0]["image"]  # PIL.Image
print(len(ds), sample.size)
```

To target a specific subfolder (e.g., hr or x4), point `data_dir` to it:

```python
ds_hr = load_dataset("imagefolder",
                     data_dir="hf://datasets/bezdarnost/DF2K-bicubic/hr",
                     #split="train"
)
ds_lrx4 = load_dataset("imagefolder",
                       data_dir="hf://datasets/bezdarnost/DF2K-bicubic/lr_x4",
                       #split="train"
)
```

### Option B — Download locally with `huggingface_hub`

```bash
pip install -U huggingface_hub
```

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="bezdarnost/DF2K-bicubic",
    repo_type="dataset",
    local_dir="./data/DF2K-bicubic",
    local_dir_use_symlinks=False  # set to False if your FS doesn't like symlinks
)
print("Downloaded to:", local_dir)
```

---

## 🗂️ Expected Structure

The mirror follows a simple, discoverable layout. Typical folders you might see:

```
DF2K-bicubic/
├─ hr/                # High-resolution images
├─ x2/                # Bicubic-downsampled LR ×2
├─ x3/                # Bicubic-downsampled LR ×3
├─ x4/                # Bicubic-downsampled LR ×4
└─ README.md          # README from maintainer
```

---

## ✅ Integrity & Reproducibility

* Files are mirrored 1:1 from the upstream source where possible.

---

## 📜 Licenses & Terms

* We **do not** change or re-license datasets. Each dataset remains under its **original license and terms**.
* Before using a mirror, review the original dataset’s license and citation policy and ensure your use case complies (research, commercial, redistribution, etc.).

---

## 🔗 How to Cite

If your work uses these mirrors, please:

1. **Cite the original dataset** (per its authors’ instructions).
2. Optionally acknowledge this mirror, e.g.:

> “We accessed DF2K via the Hugging Face mirror `bezdarnost/DF2K-bicubic`.”

---

## 🤝 Contributing / Request a Mirror

* Want another dataset mirrored?
  Open an issue with:

  * Dataset name + link
  * Preferred layout (e.g., `hr/`, `lr_x4/`)
  * Any preprocessing expectations
* Found a broken file or mismatch?
  Please include paths, expected behavior, and a minimal reproduction.
