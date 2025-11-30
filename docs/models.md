# ClarityCore Models

This document provides comprehensive information about all models available in ClarityCore, including architecture details, training presets, and usage examples.

## Quick Reference

| Model | Task | Parameters | Scales | Paper |
|-------|------|------------|--------|-------|
| [RRDBNet](#rrdbnet) | Super-Resolution | 16.7M (full) / 1M (lite) | 2×, 3×, 4×, 8× | [ESRGAN](https://arxiv.org/abs/1809.00219) |

---

## RRDBNet

**Residual-in-Residual Dense Block Network** - The generator architecture from ESRGAN.

### Overview

RRDBNet is a powerful CNN architecture for image super-resolution. It achieves excellent perceptual quality through its dense residual connections and is the backbone of many state-of-the-art SR models.

**Key Features:**
- No batch normalization (removes artifacts, enables flexible inference)
- Residual-in-Residual Dense Blocks (RRDB) for deep feature extraction
- Leaky ReLU activations throughout
- Pixel shuffle upsampling for efficient high-resolution output

### Architecture Details

```
Input (H×W×3)
    │
    ▼
┌─────────────────────────┐
│   Conv 3×3 (num_feat)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│                         │
│    RRDB × num_block     │  ◄── Main feature extraction
│                         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Conv 3×3 (num_feat)   │
└────────────┬────────────┘
             │
             + ─────────────────────┐ (Global residual)
             │                      │
             ▼                      │
┌─────────────────────────┐         │
│   Upsample (×scale)     │         │
│   (Pixel Shuffle)       │         │
└────────────┬────────────┘         │
             │                      │
             ▼                      │
┌─────────────────────────┐         │
│   Conv 3×3 (num_feat)   │ ◄───────┘
│   LeakyReLU             │
│   Conv 3×3 (3)          │
└────────────┬────────────┘
             │
             ▼
      Output (sH×sW×3)
```

### Configuration

```python
from claritycore.models import AutoConfig, AutoModel

# Default configuration
config = AutoConfig.from_name("rrdbnet", scale=4)

# Custom configuration
config = AutoConfig.from_name(
    "rrdbnet",
    scale=4,
    num_feat=64,      # Feature channels (32 for lite, 64 for full)
    num_block=23,     # RRDB blocks (6 for lite, 23 for full)
    num_grow_ch=32,   # Dense block growth channels
)

model = AutoModel.from_config(config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | 3 | Input image channels |
| `out_channels` | int | 3 | Output image channels |
| `scale` | int | 4 | Upscaling factor (2, 3, 4, or 8) |
| `num_feat` | int | 64 | Intermediate feature channels |
| `num_block` | int | 23 | Number of RRDB blocks |
| `num_grow_ch` | int | 32 | Growth channels in dense blocks |

### Variants

#### Full Model (~16.7M parameters)
Best quality, suitable for high-end applications.

```python
config = AutoConfig.from_name("rrdbnet", scale=4, num_feat=64, num_block=23)
```

#### Lite Model (~1M parameters)
Fast inference, suitable for real-time applications.

```python
config = AutoConfig.from_name("rrdbnet", scale=4, num_feat=32, num_block=6)
```

---

## Training Presets

ClarityCore provides optimized training presets for quick experimentation.

### RRDBNet Presets

<details>
<summary><b>rrdbnetx2</b> - 2× Super-Resolution (Full)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 2× |
| Features | 64 |
| Blocks | 23 |
| Parameters | ~16.7M |
| **Training** | |
| Total Steps | 400,000 |
| Batch Size | 16 |
| Patch Size | 128×128 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnetx2
```

</details>

<details>
<summary><b>rrdbnetx3</b> - 3× Super-Resolution (Full)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 3× |
| Features | 64 |
| Blocks | 23 |
| Parameters | ~16.7M |
| **Training** | |
| Total Steps | 400,000 |
| Batch Size | 16 |
| Patch Size | 192×192 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnetx3
```

</details>

<details>
<summary><b>rrdbnetx4</b> - 4× Super-Resolution (Full)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 4× |
| Features | 64 |
| Blocks | 23 |
| Parameters | ~16.7M |
| **Training** | |
| Total Steps | 400,000 |
| Batch Size | 16 |
| Patch Size | 256×256 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnetx4
```

</details>

<details>
<summary><b>rrdbnetx8</b> - 8× Super-Resolution (Full)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 8× |
| Features | 64 |
| Blocks | 23 |
| Parameters | ~16.7M |
| **Training** | |
| Total Steps | 500,000 |
| Batch Size | 8 |
| Patch Size | 512×512 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnetx8
```

</details>

<details>
<summary><b>rrdbnet-litex2</b> - 2× Super-Resolution (Lite)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 2× |
| Features | 32 |
| Blocks | 6 |
| Parameters | ~1M |
| **Training** | |
| Total Steps | 200,000 |
| Batch Size | 32 |
| Patch Size | 128×128 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnet-litex2
```

</details>

<details>
<summary><b>rrdbnet-litex3</b> - 3× Super-Resolution (Lite)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 3× |
| Features | 32 |
| Blocks | 6 |
| Parameters | ~1M |
| **Training** | |
| Total Steps | 200,000 |
| Batch Size | 32 |
| Patch Size | 192×192 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnet-litex3
```

</details>

<details>
<summary><b>rrdbnet-litex4</b> - 4× Super-Resolution (Lite)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 4× |
| Features | 32 |
| Blocks | 6 |
| Parameters | ~1M |
| **Training** | |
| Total Steps | 200,000 |
| Batch Size | 32 |
| Patch Size | 256×256 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnet-litex4
```

</details>

<details>
<summary><b>rrdbnet-litex8</b> - 8× Super-Resolution (Lite)</summary>

| Setting | Value |
|---------|-------|
| **Architecture** | |
| Scale | 8× |
| Features | 32 |
| Blocks | 6 |
| Parameters | ~1M |
| **Training** | |
| Total Steps | 250,000 |
| Batch Size | 16 |
| Patch Size | 512×512 |
| Learning Rate | 2e-4 |
| Loss | L1 |
| **Features** | |
| Mixed Precision | ✓ |
| EMA | ✗ |

```bash
claritycore rrdbnet-litex8
```

</details>

---

## Custom Training

Override preset values with command-line arguments:

```bash
# Use preset with modifications
claritycore rrdbnetx4 --batch-size 8 --steps 100000 --ema

# Full control with train command
claritycore train \
    --model rrdbnet \
    --scale 4 \
    --num-feat 64 \
    --num-block 23 \
    --data datasets/DIV2K \
    --batch-size 16 \
    --steps 400000 \
    --lr 2e-4 \
    --loss l1 \
    --amp \
    --name my_experiment
```

---

## Adding New Models

To add a new model to ClarityCore:

1. Create a new directory under `claritycore/models/`:
   ```
   claritycore/models/mymodel/
   ├── __init__.py
   ├── config.py       # Model config with training presets
   ├── architecture.py # Pure nn.Module implementation
   └── model.py        # Training wrapper (BaseModel subclass)
   ```

2. Define your config with presets in `config.py`:
   ```python
   from dataclasses import dataclass
   from claritycore.models.base import BaseConfig
   from claritycore.models.auto import register_config

   @register_config("mymodel")
   @dataclass
   class MyModelConfig(BaseConfig):
       model_type: str = "mymodel"
       # ... your config fields

   # Define training presets
   MYMODEL_PRESETS = {
       "mymodelx4": (MyModelConfig(scale=4), MyModelTrainingPreset(...)),
   }
   ```

3. Register your model in `claritycore/models/__init__.py`:
   ```python
   from claritycore.models.mymodel import MyModelConfig, MyModelModel
   ```

4. Add your presets to `claritycore/cli/presets.py`:
   ```python
   from claritycore.models.mymodel.config import MYMODEL_PRESETS
   # ... merge into all presets
   ```

---

## References

- **ESRGAN**: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", ECCV 2018 Workshop. [Paper](https://arxiv.org/abs/1809.00219)
- **Real-ESRGAN**: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCV 2021 Workshop. [Paper](https://arxiv.org/abs/2107.10833)

