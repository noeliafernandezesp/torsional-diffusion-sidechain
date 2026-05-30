# ChiGNN — Generative Modeling of Protein Side Chains via Torsional Diffusion
 
> **Master's Thesis** · Master's Degree in Bioinformatics and Biostatistics (UOC)  
> Author: **Noelia Fernández Espiñeira** · Supervisor: **Brian Jiménez García**  
> Area: Drug Design and Structural Biology · June 2026
 
🌐 [Versión en español](README.md)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeliafernandezesp/torsional-diffusion-sidechain/blob/main/TFM_script.ipynb)

---
 
## Abstract
 
**ChiGNN** is a generative AI model based on **torsional diffusion over the circular space S¹** for recovering protein side-chain conformations from a fixed backbone. The core technical contribution is the use of the **Von Mises distribution** as the noise source in the forward diffusion process, which naturally respects the periodic and circular nature of dihedral angles χ₁–χ₄, unlike standard Gaussian approaches.
 
The model was trained on **597 high-resolution protein structures** (<2.0 Å) from PDB-REDO and achieved a **rotamer recovery rate of 53.9%** (±40° tolerance) with a circular MAE for χ₁ of **56.41°**, outperforming random and modal baselines for χ₂–χ₄. Confidence calibration analysis yields a Spearman coefficient of **ρ = 0.299** (p < 0.001), demonstrating a genuine self-diagnostic signal in the model.
 
---
 
## Table of Contents
 
- [Motivation](#motivation)
- [Model Architecture](#model-architecture)
- [Project Pipeline](#project-pipeline)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Comparison with State of the Art](#comparison-with-state-of-the-art)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [License](#license)
---
 
## Motivation
 
Although tools like AlphaFold2 predict the protein backbone with high fidelity (TM-score > 0.9), the positions of side-chain dihedral angles χ₁–χ₄ remain imprecise (~70–75% recovery for χ₁). This limitation is critical because **side chains directly determine biological function**: they define enzyme active sites, mediate ligand and drug binding, and establish hydrogen bonds that stabilise the tertiary structure.
 
Classical methods (SCWRL4, Rosetta) model side-chain conformations as a **deterministic optimisation** problem, finding a single lowest-energy state without capturing the distribution of possible conformations. ChiGNN addresses this through a **probabilistic generative** approach with calibrated uncertainty.
 
---
 
## Model Architecture
 
ChiGNN is a *lite* architecture inspired by the principles of DiffPack, designed to run in cloud computing environments (Google Colab with a T4 GPU).
 
```
Input: protein graph G(V, E)
  ├── Nodes V: amino acid residues (Cα coordinates, residue type, φ/ψ angles)
  └── Edges E: spatial contacts (Cα distance threshold < 8 Å)
 
Forward diffusion process:
  χ_t = χ_0 + ε,   ε ~ Von Mises(0, κ)
 
GNN architecture (reverse process):
  ├── 4 × GCNConv with BatchNorm and residual connections
  ├── Activation function: ReLU
  └── Output head: gradient prediction ∇ log p(χ_t)
 
Reverse diffusion: DDIM adapted to the circular space S¹
Output: recovered dihedral angles χ₁, χ₂, χ₃, χ₄ per residue
```
 
**Model parameters:** 80,404 trainable parameters  
**Optimiser:** AdamW + CosineAnnealingLR  
**Training:** 50 epochs · Best checkpoint: epoch 42 (val_loss = 0.0885)
 
---
 
## Project Pipeline
 
The project is organised into three phases, all executed from a single Google Colab notebook:
 
### Phase 1 — Data Foundation and Preparation
1. Query to the RCSB REST API with quality filters: resolution <2.0 Å, X-ray crystallography, protein chains only
2. Download of re-refined structures from **PDB-REDO** (`{pid}_final.pdb`)
3. Extraction of dihedral angles χ₁–χ₄ via **Biopython** and backbone coordinates
4. Serialisation of the dataset in **PyTorch Geometric** format (`.pt`) with boolean mask `y_mask`
5. **80/10/10** protein-level split to prevent data leakage
### Phase 2 — AI Model Implementation
1. Implementation of the Von Mises forward process with **py3Dmol** visualisation
2. Design and training of the **GNN** (ChiGNN) for the reverse diffusion process
3. Conformation recovery and hyperparameter tuning
### Phase 3 — Biostatistical Evaluation and Benchmarking
1. Computation of **circular MAE** and rotamer recovery rate (±20° and ±40°)
2. Rotameric distribution analysis via **Rose Plots**
3. Confidence calibration: Spearman correlation between Von Mises circular variance and actual MAE
4. Benchmarking against SCWRL4 and random/modal baselines
The notebook includes a **pipeline control panel** with boolean flags to determine which phases are re-executed or loaded from Google Drive, minimising computation time in subsequent sessions.
 
---
 
## Results
 
| Metric | ChiGNN | Modal baseline | Random baseline |
|---|---|---|---|
| Circular MAE χ₁ | **56.41°** [95% CI: 55.63°–57.15°] | ~60–65° | ~90° |
| Rotamer recovery χ₁ (±40°) | **53.9%** | ~47% | ~33% |
| Rotamer recovery χ₁ (±20°) | ~28% | — | — |
| Calibration (Spearman ρ) | **0.299** (p < 0.001) | — | — |
 
**Rose plots** confirm that the model qualitatively reproduces the characteristic trimodal rotameric distribution (g⁻ ≈ −60°, t ≈ 180°, g⁺ ≈ +60°).
 
> **Note:** SCWRL4 achieves ~83% recovery for χ₁. The current performance gap is primarily attributed to dataset size (597 proteins vs. tens of thousands) and the non-equivariant architecture.
 
---
 
## Repository Structure
 
```
torsional-diffusion-sidechain/
│
├── TFM_script.ipynb              # Main notebook (Phases 1, 2 and 3)
├── TFM_script.py                 # Python version of the script
├── requirements.txt              # Project dependencies
├── README.md                     # Spanish version
├── README_EN.md                  # This file
├── LICENSE                       # MIT licence
└── TFM_Noelia_Fernandez.pdf      # Full thesis document
```
 
> Data files (`.pt`, `.csv`) and the model checkpoint are stored on Google Drive due to their size. See the [Dataset](#dataset) section for download instructions.
 
---
 
## Installation and Requirements
 
The project is designed to run on **Google Colab with GPU** (T4 or better). Dependencies are installed automatically from Block 2 of the notebook.
 
### System requirements
- Python ≥ 3.9
- CUDA ≥ 11.x (GPU required; training is not feasible on CPU)
- Google Colab (recommended) or a local environment with a compatible GPU
### Manual installation (local environment)
 
```bash
# Install base dependencies
pip install biopython py3Dmol rdkit
 
# PyTorch Geometric (adjust versions to match your environment)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
 
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install torch-sparse  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install torch-geometric
```
 
### Technology stack
 
| Library | Purpose |
|---|---|
| PyTorch + CUDA | Training backend and diffusion |
| PyTorch Geometric | GNN graph construction and message passing |
| Biopython | PDB structure parsing and dihedral angle computation |
| NumPy / Pandas | Data processing and metrics |
| Matplotlib | Rose plots, learning curves, histograms |
| py3Dmol | Interactive 3D protein visualisation |
| RDKit | Auxiliary computational chemistry |
 
---
 
## Usage
 
### Option A — Google Colab (recommended)
 
1. Open `TFM_script.ipynb` in Google Colab
2. Enable the GPU: `Runtime → Change runtime type → T4 GPU`
3. Mount your Google Drive when prompted
4. **First run:** in Block 0, set `FORCE_ALL = True` and run all cells
5. **Subsequent runs:** keep all flags as `False` to load from Drive (~30 sec)
### Option B — Inference on a new structure
 
```python
from Bio.PDB import PDBParser
import torch
from model import ChiGNN, run_reverse_diffusion
 
# Load structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "my_protein.pdb")
 
# Load model
model = ChiGNN()
model.load_state_dict(torch.load("models/chignn_best.pt"))
# Repository: https://github.com/noeliafernandezesp/torsional-diffusion-sidechain
model.eval()
 
# Recover side chains
chi_pred = run_reverse_diffusion(model, structure, n_steps=100)
print(chi_pred)  # predicted χ₁–χ₄ angles per residue
```
 
### Pipeline control panel
 
Block 0 of the notebook controls which phases are re-executed:
 
```python
FORCE_ALL        = False  # True = regenerate everything from scratch
FORCE_DOWNLOAD   = False  # Re-download PDBs from PDB-REDO
FORCE_EXTRACTION = False  # Re-extract chi angles and coordinates
FORCE_DATASET    = False  # Rebuild graphs and train/val/test split
FORCE_TRAINING   = False  # Re-train the GNN
FORCE_INFERENCE  = False  # Re-generate test set predictions
```
 
The cascade logic ensures consistency: re-extracting the dataset automatically triggers re-training and re-inference.
 
---
 
## Dataset
 
The dataset was built from **PDB-REDO**, a database of crystallographic structures re-refined with REFMAC5, providing greater accuracy in atomic coordinates and bond geometry compared to the original PDB entries.
 
**Selection criteria:**
- Resolution < 2.0 Å (X-ray crystallography)
- Protein chains only (no DNA/RNA)
- Availability in PDB-REDO (`{pid}_final.pdb`)
**Curated dataset statistics:**
 
| Parameter | Value |
|---|---|
| Total proteins | 597 |
| Total residues | 156,407 |
| Train / val / test split | 80% / 10% / 10% (protein-level) |
| χ₁ distribution | Confirmed trimodal (g⁻, t, g⁺) |
| Format | `.csv` (human-readable) + `.pt` (PyTorch Geometric) |
| Download date | 2 April 2026 |
 
PDB file download and processing are handled automatically by the notebook (Blocks 5–9). Processed files are stored on Google Drive under `MyDrive/TFM_SideChain/`.
 
---
 
## Comparison with State of the Art
 
| Method | Type | χ₁ Recovery (±40°) | Uncertainty |
|---|---|---|---|
| **ChiGNN** (this work) | Torsional diffusion (GNN) | 53.9% | ✅ Calibrated (Von Mises) |
| SCWRL4 | Rotamer library | ~83% | ❌ No |
| Rosetta FastRelax | Force field | ~85% | ❌ No |
| AlphaFold2 | Predictive deep learning | ~70–75% | ❌ No |
| AlphaFold3 | All-atom diffusion | >90% | Partial |
| DiffPack | Torsional diffusion | ~85% | ✅ Yes |
 
The **key advantage of ChiGNN** over deterministic methods lies not in point accuracy but in **uncertainty calibration**: the Von Mises circular variance of the reverse process samples correlates significantly with the actual prediction error (ρ = 0.299, p < 0.001), enabling identification of low-confidence predictions without additional experimental information.
 
---
 
## Limitations and Future Work
 
### Current limitations
- **Small dataset:** 597 proteins versus the tens of thousands used by DiffPack or AlphaFold3
- **Non-equivariant architecture:** GCNConv is not invariant to rotations and translations of the reference frame; an EGNN or SE(3)-equivariant architecture would substantially improve accuracy
- **Evaluation constraints:** the absence of a locally installed SCWRL4 in the Colab environment prevents direct comparison on the same test set
### Future directions
1. **Dataset scaling:** expand to ≥10,000 PDB-REDO proteins with balancing by amino acid type
2. **Equivariant architectures:** integrate EGNN, GVP or ProteinMPNN as the denoiser backbone
3. **Multi-angle diffusion:** model the joint distribution P(χ₁, χ₂, χ₃, χ₄) instead of treating angles independently
4. **External validation:** benchmarking on CASP15 and the standard SCWRL4 test set
5. **Iterative refinement:** combine ChiGNN with a lightweight force field as a post-processing step
---
 
## Citation
 
If you use this work in your research, please cite:
 
```bibtex
@mastersthesis{fernandez2026chignn,
  author  = {Fernández Espiñeira, Noelia},
  title   = {Modelado Generativo de Cadenas Laterales de Proteínas
             mediante Difusión Torsional en Espacios No Euclidianos},
  school  = {Universitat Oberta de Catalunya (UOC)},
  year    = {2026},
  month   = {June},
  program = {Master's Degree in Bioinformatics and Biostatistics},
  area    = {Drug Design and Structural Biology},
  advisor = {Jiménez García, Brian},
  url     = {https://github.com/noeliafernandezesp/torsional-diffusion-sidechain}
}
```
 
---
 
## License
 
© 2026 Noelia Fernández Espiñeira. All rights reserved.
 
The source code in this repository is distributed under the **MIT licence** for academic and research use. Reproduction of the thesis document, in whole or in part, by any means is prohibited without the explicit written authorisation of the author.
 
---
 
*Developed as a Master's Thesis for the Master's Degree in Bioinformatics and Biostatistics at UOC, in the area of Drug Design and Structural Biology.*
