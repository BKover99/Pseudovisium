# 🛑 Pseudovisium
Pseudovisium is a powerful Python software package designed to streamline the processing and analysis of imaging-based spatial transcriptomics data. By leveraging hexagonal binning, Pseudovisium enables efficient compression and visualization of spatial data, making it easier to extract meaningful insights from your experiments.

## 🚀 Key Features
1. generate_pv: Effortless Data Compression 📊
The generate_pv command takes your spatial transcriptomics data in CSV format and performs hexagonal binning to compress the data while preserving spatial information. It calculates hexagon counts and cell counts, and creates a well-structured output directory with all the necessary files for downstream analysis.

2. merge_visium: Seamless Data Integration 🧩
With the merge_visium command, you can easily merge multiple Pseudovisium or Visium format files. This feature allows you to combine data from different datasets, stitch images together, and generate a merged output directory. Pseudovisium makes it effortless to integrate data from multiple spatial transcriptomics experiments, enabling comprehensive analysis across datasets.

3. generate_qc_report: Comprehensive Quality Control 📈
Pseudovisium's generate_qc_report command generates a detailed quality control (QC) report for your Pseudovisium output. It calculates a wide range of metrics, including the number of hexagons with a minimum count threshold, the number of genes present in a certain percentage of hexagons, and the median counts and features per hexagon. The report also showcases interactive hexagon plots for selected genes and provides comparison plots between different datasets, allowing you to assess the quality and consistency of your data.

## 🎯 Flexibility and Compatibility
Pseudovisium is designed to be flexible and compatible with various spatial transcriptomics technologies. It supports data from different platforms and offers customizable output formats, ensuring seamless integration with other analysis tools and workflows.
Technologies tried include:

#### Vizgen
#### Xenium
#### CosMx
#### Curio
#### seqFISH
#### VisiumHD


🚀 Get Started with Pseudovisium
To start using Pseudovisium, simply install the package and explore the documentation and examples provided. Pseudovisium is open-source and actively maintained, ensuring continuous improvements and support for the spatial transcriptomics community.

Pseudovisium is available on PyPI and can be easily installed using pip:

###### pip install Pseudovisium

For more information and the latest version, visit the Pseudovisium PyPI page https://pypi.org/project/Pseudovisium/.


### Goals

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
