# MASS: Mixture-of-Experts for Adaptive Semantic Specialization

This repository contains the implementation of **MASS (Mixture-of-Experts for Adaptive Semantic Specialization)**, a novel approach that combines Mixture of Experts (MoE) with adaptive expert expansion for both **Language** and **Vision** tasks. MASS dynamically adjusts the number of experts based on gradient dynamics, enabling efficient domain generalization and specialization.

## 🚀 Key Features

- **Adaptive Expert Expansion**: Dynamically expands experts during training based on gradient-driven semantic drift signals
- **MinTau Routing**: Novel adaptive gate mechanism that selects experts based on cumulative routing mass
- **Experiments**: Specialized for both language (GLUE tasks) and vision (domain generalization) tasks

---

## 📁 Repository Structure

```
├── Language/                    # Language tasks (GLUE benchmarks)
│   ├── search_glue_no_trainer_mass.py   # Main training script for language
│   ├── moe_utils_mass.py               # MoE utilities for language models
│   ├── scripts_mass/                   # Experiment scripts for GLUE tasks
│   └── requirements.txt                # Language dependencies
├── Vision/                      # Vision tasks (Domain generalization)
│   ├── domainbed/                      # Domain generalization framework
│   ├── ├── vision_transformers.py      
│   │   ├── algorithms_mass.py          # MASS algorithm implementation
│   │   ├── scripts/train_mass.py       # Main training script for vision
│   │   └── moe_utils.py                # Vision Transformer MoE conversion
│   ├── scripts_mass/                   # Experiment scripts for vision datasets
│   └── requirements.txt                # Vision dependencies
└── tutel/                       # Tutel MoE library with MASS extensions
    ├── tutel/gates/mintau.py              # MinTau adaptive gating mechanism
    └── tutel/impls/moe_layer_mass.py      # MASS-enabled MoE layer with expert expansion

```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU
- PyTorch 1.12+

### Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd mass

# Install Tutel library with MASS extensions
cd tutel
python setup.py clean --all
pip install -e .
```

### Install Dependencies

**For Language Tasks:**
```bash
cd Language
pip install -r requirements.txt
```

**For Vision Tasks:**
```bash
cd Vision
pip install -r requirements.txt
```

---

## 🎯 Language Tasks (GLUE Benchmarks)

### Supported Tasks
- **MNLI** (Multi-Genre Natural Language Inference)
- **CoLA** (Corpus of Linguistic Acceptability)
- **RTE** (Recognizing Textual Entailment)
- **QNLI** (Question Natural Language Inference)
- **MRPC** (Microsoft Research Paraphrase Corpus)

### Quick Start

```bash
# Run CoLA with MASS
bash Language/scripts_mass/cola.sh

# Run RTE with MASS
bash Language/scripts_mass/rte.sh

# Run MNLI with MASS
bash Language/scripts_mass/mnli.sh

# Run QNLI with MASS
bash Language/scripts_mass/qnli.sh

# Run MRPC with MASS
bash Language/scripts_mass/mrpc.sh
```

---

## 👁️ Vision Tasks (Domain Generalization)

### Supported Datasets
- **PACS** (Photo, Art, Cartoon, Sketch)
- **VLCS** (VOC2007, LabelMe, Caltech101, SUN09)
- **OfficeHome** (Art, Clipart, Product, Real)
- **TerraIncognita** (Location-based terrain classification)

### Datasets
```bash
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

### Quick Start for Training

```bash
cd Vision

# Run PACS with MASS
bash scripts_mass/run_pacs.sh

# Run VLCS with MASS
bash scripts_mass/run_vlcs.sh

# Run OfficeHome with MASS
bash scripts_mass/run_office.sh

# Run TerraIncognita with MASS
bash scripts_mass/run_terra.sh
```

### Evaluation

```bash
python3 -m domainbed.scripts.collect_results --input_dir=${output_dir}
```

---

<!-- **Happy experimenting with MASS! 🚀**  -->