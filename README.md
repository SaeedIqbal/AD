# Resolving the Non-Identifiability of Recurrence Deficits in Alzheimer’s Disease through Geometric, Modulatory, and Quantization-Sensitive Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![M/EEG](https://img.shields.io/badge/M%2FEEG-Neuroscience-orange)

## 📌 Problem Statement & Research Gap

**Current approaches to characterizing disrupted temporal organization in Alzheimer's disease (AD) rely on symbolic representations of dynamic functional connectivity (dFC) derived from a fixed, group-level set of meta-states and a hard-assignment quantization of continuous connectivity trajectories.** This framework assumes that reduced recurrence in the resulting symbolic sequences directly reflects a loss of intrinsic brain dynamics.

However, this assumption conflates **three distinct dynamical phenomena**:

1. **True recurrence erosion**: A reduction in temporal self-similarity due to neurodegeneration-induced degradation of an underlying low-dimensional attractor
2. **Causal nonstationarity**: The presence of a time-varying exogenous or endogenous driver (e.g., vigilance, autonomic tone) that creates structured but non-recurrent dynamics
3. **Quantization artifacts**: Boundary crossings between Voronoi cells due to hard partitioning of blurred meta-state boundaries—as is hypothesized in AD

> **The central unresolved problem**: Given only the observed symbolic sequence \( S \), it is mathematically ill-posed to infer whether a reduction in \( \rho_S(\tau) \) in AD arises from (i) loss of intrinsic recurrence, (ii) deterministic nonstationarity, or (iii) artifacts of the quantization map.

This non-identifiability undermines the biological validity of recurrence-based biomarkers and obscures distinct pathophysiological processes with divergent clinical implications.

## 🚀 Main Contributions

Our proposed framework **resolves this non-identifiability** by disentangling three distinct mechanisms underlying recurrence loss:

### 1. **Attractor Stability Criterion** 
- **Geometric recurrence index** \( \widehat{\mathcal{R}} \) as a direct probe of attractor integrity
- Operates natively on continuous IAC trajectories without symbolic discretization
- **Outcome**: AD shows significantly reduced \( \widehat{\mathcal{R}} \) (p < 0.001), confirming true neurodegenerative collapse

### 2. **External Modulation Detection**
- **Sliding-window conditional independence testing** to detect nonstationary coupling to physiological drivers
- Global modulation index \( \Gamma \) quantifies adaptive brain-body interactions  
- **Outcome**: MCI shows elevated \( \Gamma \) (p < 0.01), suggesting compensatory dynamics

### 3. **Quantization Sensitivity Analysis**
- **Perturbation-based analysis** of meta-state centroid uncertainty
- Relative sensitivity \( \widetilde{\mathcal{Q}} \) identifies representation artifacts
- **Outcome**: AD shows high \( \widetilde{\mathcal{Q}} \) vs FTD (p < 0.001), confirming AD-specific boundary blurring

## 📊 Datasets & Experimental Details

Our framework is validated across **four open-source datasets** spanning the AD spectrum:

| Dataset | Samples | Clinical Groups | Channels | Modality | Sampling Rate | Auxiliary Signals |
|---------|---------|----------------|----------|----------|---------------|-------------------|
| **OpenNeuro ds004504** | 88 | 29 HC, 36 AD, 23 FTD | 19 | EEG | 500 Hz | ECG (60%) |
| **BioFIND** | 324 | 154 HC, 170 MCI | 306 | MEG | 1000 Hz | ECG (100%) |
| **Mendeley Olfactory** | 35 | 15 HC, 7 MCI, 13 AD | 4 | EEG | 256 Hz | None |
| **PREVENT-AD** | 350+ | Mixed High-Risk | 32/64 | EEG | 256/500 Hz | ECG (100%) |

### Dataset References
- **OpenNeuro ds004504**: [OpenNeuro Dataset ds004504](https://openneuro.org/datasets/ds004504)
- **BioFIND**: [BioFIND Study](https://www.michaeljfox.org/biofind)
- **Mendeley Olfactory**: [Mendeley Data Olfactory EEG](https://doi.org/10.17632/5c4v7x9c3f.1)
- **PREVENT-AD**: [PREVENT-AD Open Data](https://prevent-alzheimer.ca/en/research/open-data)

## 📈 Key Results & Outcomes

### Quantitative Validation Results

| Mechanism | OpenNeuro (AD) | BioFIND (MCI) | Mendeley (AD) | PREVENT-AD (MCI) |
|-----------|----------------|---------------|---------------|------------------|
| **Attractor Stability \( \widehat{\mathcal{R}} \)** | 1.25 ± 0.28*** | 1.65 ± 0.35*** | 0.95 ± 0.30*** | 1.85 ± 0.32*** |
| **Modulation \( \Gamma \)** | 0.22 ± 0.11 | 0.86 ± 0.11*** | N/A | 0.75 ± 0.12*** |
| **Quantization Sensitivity \( \widetilde{\mathcal{Q}} \)** | 0.68 ± 0.14*** | 0.42 ± 0.12 | 0.92 ± 0.15*** | 0.38 ± 0.11 |

***p < 0.001 vs HC; **p < 0.01 vs HC

### Clinical Correlations

| Metric | Correlation with MMSE | p-value | Interpretation |
|--------|----------------------|---------|----------------|
| **Attractor Stability \( \widehat{\mathcal{R}} \)** | ρ = 0.68 | < 1e-6 | Strong predictor of cognitive integrity |
| **Quantization Sensitivity \( \widetilde{\mathcal{Q}} \)** | ρ = -0.52 | < 0.001 | AD-specific boundary blurring |
| **Modulation \( \Gamma \) (MCI only)** | ρ = 0.41 | < 0.01 | Compensatory adaptive mechanism |

### Key Findings
1. **AD is characterized by concurrent attractor erosion and elevated quantization sensitivity**
2. **MCI is defined by significantly increased nonstationary modulation, not degradation**
3. **Quantization sensitivity differentiates AD from FTD, confirming AD-specific pathology**
4. **Modulation positively correlates with cognitive scores in MCI, suggesting compensation**

## 🏗️ Repository Structure

```
ad-metastate-dynamics/
├── README.md
├── requirements.txt
├── config/
│   └── dataset_paths.yaml
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openneuro.py
│   │   ├── biofind.py
│   │   ├── mendeley.py
│   │   └── prevent_ad.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── iac.py
│   │   ├── recurrence.py
│   │   ├── modulation.py
│   │   ├── quantization.py
│   │   └── attractor.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── source_reconstruction.py
│       └── statistical_tests.py
├── notebooks/
│   └── validation_synthetic.ipynb
├── scripts/
│   ├── run_preprocessing.py
│   └── run_analysis.py
└── results/
    ├── figures/
    └── metrics/
```

## ⚙️ Configuration (dataset_paths.yaml)

```yaml
datasets:
  OpenNeuro_ds004504:
    path: "/home/phd/datasets/OpenNeuro_ds004504"
    type: "eeg"
    format: "eeglab"
    channels: 19
    sfreq: 500.0
    montage: "10-20"
    
  BioFIND:
    path: "/home/phd/datasets/BioFIND"
    type: "meg" 
    format: "fif"
    channels: 306
    sfreq: 1000.0
    montage: "MEG"
    
  Mendeley_Olfactory:
    path: "/home/phd/datasets/Mendeley_Olfactory"
    type: "eeg"
    format: "mat"
    channels: 4
    sfreq: 256.0
    montage: "4-channel"
    
  PREVENT_AD:
    path: "/home/phd/datasets/PREVENT_AD"
    type: "eeg"
    format: "brainstorm"
    channels: [32, 64]
    sfreq: [256.0, 500.0]
    montage: "extended-10-20"
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/ad-metastate-dynamics.git
cd ad-metastate-dynamics
pip install -r requirements.txt
```

### Data Preparation
1. Download the four datasets to `/home/phd/datasets/`
2. Update `config/dataset_paths.yaml` with your local paths

### Run Preprocessing
```bash
python scripts/run_preprocessing.py --config config/dataset_paths.yaml
```

### Run Analysis
```bash
python scripts/run_analysis.py --input-dir /home/phd/results --output-dir /home/phd/results
```

### Validate with Synthetic Data
```bash
jupyter notebook notebooks/validation_synthetic.ipynb
```

## 📋 Requirements

```txt
numpy>=1.21.0
scipy>=1.7.0
mne>=1.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pyyaml>=6.0
pandas>=1.3.0
h5py>=3.6.0
```

> **Note**: Install MNE-Python for EEG/MEG processing: `pip install mne`

## 📚 Methodological Comparison

| Approach | Attractor Erosion | Nonstationary Modulation | Quantization Artifacts | Mechanism Disentanglement |
|----------|-------------------|--------------------------|------------------------|---------------------------|
| **Traditional Symbolic Recurrence** | ❌ Confounded | ❌ Confounded | ❌ Confounded | ❌ No |
| **Continuous Recurrence Only** | ✅ Detected | ❌ Confounded | ✅ Avoided | ❌ Partial |
| **Modulation Detection Only** | ❌ Missed | ✅ Detected | ❌ Confounded | ❌ Partial |
| **Our Proposed Framework** | ✅ Isolated | ✅ Isolated | ✅ Isolated | ✅ Complete |

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{your2025disentangling,
  title={Disentangling Attractor Degradation, Nonstationary Modulation, and Quantization Artifacts in Alzheimer's Disease Meta-State Dynamics},
  author={Saeed Iqbal},
  journal={NeuroImage},
  year={2025},
  publisher={Elsevier}
}
```

## 📧 Contact

For questions or collaboration opportunities, please contact:  
**Saeed Iqbal** - saeed.iqbal@szu.edu.cn  
College of Mechatronics and Control Engineering, Shenzhen University, China

---

**This repository provides a complete, reproducible implementation of a novel framework that transforms the interpretation of "lost recurrence" in Alzheimer's disease from a monolithic marker of decay into a mechanistically rich signature with distinct clinical and therapeutic implications.**
