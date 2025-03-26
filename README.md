# OCR Visual Fidelity Verification

A tool for verifying and improving the quality of OCR results from degraded machine-typed documents. This project focuses on post-processing OCR data to reduce the need for manual validation by identifying potential errors and providing quality scores.

## Features

- **JSON Parsing**: Parse Azure Document Intelligence JSON format maintaining original structure
- **Quality Verification**: Multiple methods to verify OCR quality without ground truth
- **Visualization**: Overlay quality indicators on original document

## Installation

```bash
# Clone the repository
git clone https://github.com/mengqvist/ocr_visual_fidelity_verification.git
cd ocr_visual_fidelity_verification
```

Use the `environment-dev.yml` file to create a conda environment and install the dependencies. This assumes that you have conda installed (https://docs.conda.io/en/latest/miniconda.html).

```bash
# Create a conda environment
conda env create -f environment-dev.yml

# Activate the conda environment
conda activate ocr_quality_dev
```

Then install the package with pip.

```bash
pip install -e .
```




## Repository Structure

The repository is structured as follows:

```
ocr-verification/
├── README.md
├── environment-dev.yml
├── setup.py
├── data/
│   ├── raw_json/
│   ├── processed_json/
│   └── pdfs/
├── fonts/
│   ├── Courier.ttf
│   ├── Elite.ttf
│   ├── Pica.ttf
│   └── ....
├── vfv/
│   ├── __init__.py
│   ├── helpers.py
│   ├── interactive_viewer.py
│   ├── json_parser.py
│   ├── viewer.py
│   └── words.py
├── logs/
└── tests/
    ├── test_parser.py
    └── test_words.py
```

