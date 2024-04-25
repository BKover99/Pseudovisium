# 🛑 Pseudovisium
Pseudovisium is a Python software package designed to democratize the analysis of spatial transcriptomics data. By leveraging hexagonal binning, Pseudovisium enables efficient compression and visualization of spatial data, making exploratory analysis and quality control faster and more memory efficient. The goal of this tool is not to increase accuracy, but to make spatial data analysis more accessible, regardless of computing environment. Additionally, this package facilitates simulating low-res/Visium spatial experiments both for practical (e.g. experimental design) and theoretical (e.g. studying the impact of resolution) purposes.

## 🚀 Key Features
### 1. generate_pv:   Data Compression 📊

The generate_pv command takes your spatial transcriptomics data in CSV format and performs hexagonal binning to compress the data while preserving spatial information. It calculates hexagon counts and cell counts, and creates a well-structured output directory with all the necessary files for downstream analysis.


### 2. qc:   Quality Control 📈

Pseudovisium's qc command generates a detailed quality control (QC) report for a set of Pseudovisium/Visium formatted replicates. It calculates a wide range of metrics, including the number of hexagons with a minimum count threshold, the number of genes present in a certain percentage of hexagons, and the median counts and features per hexagon. The report also showcases hexagon plots for selected genes and provides comparison plots between different datasets, allowing you to assess the quality and consistency of your data.

### 3. merge:   Data Merging 🧩

With the merge command, you can easily merge multiple Pseudovisium or Visium format files. This feature allows you to combine data from different datasets, merge images together, and generate a merged output directory in the Pseudovisium/Visium format. Pseudovisium makes it effortless to merge data from multiple spatial transcriptomics experiments, enabling comprehensive analysis across datasets.


## 🎯 Flexibility and Compatibility

Pseudovisium is designed to be flexible and compatible with various spatial transcriptomics technologies. It supports data from different platforms and offers customizable output formats, ensuring seamless integration with other analysis tools and workflows. Input files range from transcripts.csv, to .h5 as well as 10X feature-barcode-matrix directories.
Technologies tried include:

#### Vizgen
#### Xenium
#### CosMx
#### Curio
#### seqFISH
#### VisiumHD


## 🚀 Get Started with Pseudovisium

To start using Pseudovisium, simply install the package and explore the documentation and examples provided. Pseudovisium is open-source and actively maintained, ensuring continuous improvements and support for the spatial transcriptomics community.

Pseudovisium is available on PyPI and can be easily installed using pip:

###### pip install Pseudovisium

For more information and the latest version, visit the Pseudovisium PyPI page https://pypi.org/project/Pseudovisium/.

## Examples
See the example Google Colab on converting 10X Xenium Mouse pup data to Pseudovisium format:
https://github.com/BKover99/Pseudovisium/blob/main/pseudovisium_mouse_pup.ipynb

## Goals

- [x] Releasing the repo with minimal working capabilities - Apr 20, 2024
- [ ] Adding supporting Colab notebooks
- [ ] Releasing all data used in this work on OneDrive
- [ ] Tidying up all code
- [ ] Posting pre-print to Biorxiv
- [ ] Publication of peer-reviewed research article

### Author
Bence Kover
https://twitter.com/kover_bence
https://www.linkedin.com/in/ben-kover/
