# CE-ProtoNet: Chaos-Embedded Prototypical Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DirectML-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
**CE-ProtoNet** is a novel few-shot learning algorithm designed specifically for medical image classification. It enhances a standard Prototypical Network (using a ResNet-18 backbone) by incorporating a **Logistic Chaos** module. This module injects dynamic, deterministic chaos noise directly into the support set embeddings during the training phase, artificially simulating complex intra-class variance and drastically improving generalization on extremely small datasets.

This repository provides the official implementation, natively optimized for the **PyTorch DirectML** backend, allowing it to leverage any DirectX 12 compatible GPU on Windows (AMD, Intel, or NVIDIA).

## Key Features
- **Medical ProtoNet Architecture**: Pre-trained ResNet-18 adapted for few-shot metric learning with L2 normalization and cosine similarity.
- **Logistic Chaos Module**: Dynamic logistic-map noise injection for robust support-set regularization.
- **Hardware Agnostic (Windows)**: Fully leverages `torch-directml` for GPU acceleration without requiring CUDA.
- **Episodic Task Sampler**: Built-in support for N-way K-shot medical episode sampling.

---

## Installation

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Environment
**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Availability
To ensure your hardware and drivers are correctly configured for PyTorch DirectML, run:
```python
import torch_directml
print(f"DirectML Device Available: {torch_directml.device_name(0)}")
```

---

## Usage

CE-ProtoNet is designed to be easily configurable directly from the command line.

### Basic Training
To train the model on a dataset, simply pass the path to your dataset (which must contain `train` and `test` subdirectories):

```bash
python ChaosProtoNet.py --data_path "D:\Path\To\BrainTumorDataset"
```

### Advanced Configuration (Few-Shot Settings)
You can customize the N-way K-shot parameters, learning rates, epochs, and the intensity of the Chaos module:

```bash
python ChaosProtoNet.py \
    --data_path "D:\Path\To\BrainTumorDataset" \
    --n_way 5 \
    --k_shot 1 \
    --query 15 \
    --epochs 100 \
    --chaos_intensity 0.20 \
    --lr 0.0001 \
    --output_model "ce_protonet_5way_1shot.pth"
```

### Full Command-Line Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | **Required** | Path to the dataset directory containing 'train' and 'test' folders. |
| `--epochs` | `50` | Number of training epochs. |
| `--episodes_train` | `500` | Number of episodes per training epoch. |
| `--episodes_val` | `600` | Number of episodes per validation epoch. |
| `--n_way` | `4` | N-way classification (number of classes per episode). |
| `--k_shot` | `5` | K-shot (number of support samples per class). |
| `--query` | `15` | Number of query samples per class. |
| `--lr` | `3e-4` | Learning rate. |
| `--weight_decay` | `1e-4` | Weight decay parameter for AdamW. |
| `--chaos_intensity` | `0.18` | Intensity parameter for the Logistic Chaos module. |
| `--output_model` | `ce_protonet_best.pth` | Filename to save the best model weights. |

---

## Citation
If you use CE-ProtoNet in your research, please cite our algorithm:

```bibtex
@article{ce_protonet_2026,
  title={CE-ProtoNet: Chaos-Embedded Prototypical Networks for Medical Image Classification},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
