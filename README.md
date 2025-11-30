[![GitHub Repo stars](https://img.shields.io/github/stars/Bezdarnost/claritycore?style=social)](https://github.com/Bezdarnost/claritycore)
[![Downloads](https://static.pepy.tech/badge/claritycore)](https://pepy.tech/project/claritycore)
[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Bezdarnost/claritycore/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/claritycore.svg)](https://pypi.org/project/claritycore/)
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue">
<a href="https://discord.gg/Zw2kFTruD5"><img alt="Discord" src="https://img.shields.io/discord/1426554369750470658?color=7289da&logo=discord&logoColor=white"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Bezdarnost/claritycore/main/docs/assets/logo_full.png">
    <img alt="ClarityCore" src="https://raw.githubusercontent.com/Bezdarnost/claritycore/main/docs/assets/logo_full.png" width=100%>
  </picture>
</p>

<h3 align="center">
[!!!WORK IN PROGRESS!!!]Easy and fast low-level vision for everyone
</h3>

---

[!!!WORK IN PROGRESS!!!]Next-generation Open Source toolkit for low-level vision. Engineered for state-of-the-art performance in image and video pixel2pixel tasks, including Super-Resolution, Denoising, Deblurring, and more.

<div align="center">
  
If you find ClarityCore useful, please consider giving it a star ⭐ and **[Join Discord](https://discord.gg/Zw2kFTruD5)**!

</div>

---

*Latest News* 🚀
- I am very busy at my current position. I want to continue this project in ~January
- **[2025/11]** Major refactoring: unified dataset, preset system, Rich CLI
- **[2025/10]** Project start

---

## 🚀 Quick Start

### Installation

```bash
pip install claritycore
```

Or install from source:

```bash
git clone https://github.com/Bezdarnost/claritycore.git
cd claritycore
pip install -e .
```

### Training

Train a 4x super-resolution model with one command:

```bash
# List available presets
claritycore list

# Train with a preset
claritycore train rrdbnetx4

# Train with custom settings
claritycore train rrdbnetx4 --steps 100000 --batch-size 8
```

### Python API

```python
from claritycore import AutoConfig, AutoModel, Trainer
from claritycore.data import Pixel2PixelDataset, DatasetConfig

# Create model
config = AutoConfig.from_name("rrdbnet", scale=4)
model = AutoModel.from_config(config)

# Create dataset
data_config = DatasetConfig(target_dir="data/HR", scale=4)
dataset = Pixel2PixelDataset(data_config)

# Train
trainer = Trainer(model, train_loader, optimizer, training_config)
trainer.train()
```

---

## 📦 Available Models

| Model | Task | Parameters | Scales |
|-------|------|------------|--------|
| RRDBNet | Super-Resolution | 16.7M (full) / 1M (lite) | 2×, 3×, 4×, 8× |

See [docs/models.md](docs/models.md) for detailed documentation.

---

## 📁 Dataset Structure

ClarityCore expects datasets in this structure:

```
datasets/DIV2K/
├── hr/           # Target (high-resolution) images
├── x2/           # Input images for 2x SR
├── x3/           # Input images for 3x SR
└── x4/           # Input images for 4x SR
```

The `Pixel2PixelDataset` automatically:
- Detects filename suffix patterns (e.g., `0001x4.png`)
- Generates input images on-the-fly if no input directory exists
- Supports flexible normalization ([0,1], [-1,1], or custom)

---

## 🛠️ CLI Reference

```bash
# Show help
claritycore --help

# List available training presets
claritycore list

# Train with preset
claritycore train <preset> [options]

# Examples
claritycore train rrdbnetx4                    # Full RRDBNet, 4x SR
claritycore train rrdbnet-litex4               # Lightweight variant
claritycore train rrdbnetx4 --steps 50000      # Custom steps
claritycore train rrdbnetx4 --batch-size 8     # Custom batch size

# Advanced training (full control)
claritycore train --model rrdbnet --scale 4 --data path/to/data
```

---

## 📊 Presets

| Preset | Model | Scale | Config | Steps |
|--------|-------|-------|--------|-------|
| `rrdbnetx2` | RRDBNet | 2× | 64 feat, 23 blocks | 400K |
| `rrdbnetx4` | RRDBNet | 4× | 64 feat, 23 blocks | 400K |
| `rrdbnet-litex4` | RRDBNet | 4× | 32 feat, 6 blocks | 200K |

Run `claritycore list` for all available presets.


## 📖 Citation

If you use ClarityCore in your research, please cite:

```bibtex
@software{ClarityCore_2025,
  author = {Urumbekov, Aman},
  license = {Apache-2.0},
  title = {{ClarityCore}},
  url = {https://github.com/Bezdarnost/claritycore},
  year = {2025}
}
```

Urumbekov, A. (2025). ClarityCore: A toolkit for low-level vision. GitHub. https://github.com/bezdarnost/claritycore

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bezdarnost/claritycore&type=Date)](https://www.star-history.com/#bezdarnost/claritycore&Date)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE).
