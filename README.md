# 🛑 Pseudovisium
Pseudovisium is a Python software package designed to democratize the analysis of spatial transcriptomics data. By leveraging hexagonal binning, Pseudovisium enables efficient compression and visualization of spatial data, making exploratory analysis and quality control faster and more memory efficient. The goal of this tool is not to increase accuracy, but to make spatial data analysis more accessible, regardless of computing environment. Additionally, this package facilitates simulating low-res/Visium spatial experiments both for practical (e.g. experimental design) and theoretical (e.g. studying the impact of resolution) purposes.

## 🚀 Key Features
### 1. generate_pv:   Data Compression 📊

The generate_pv command takes your spatial transcriptomics data in CSV format and performs hexagonal binning to compress the data while preserving spatial information. It calculates hexagon counts and cell counts, and creates a well-structured output directory with all the necessary files for downstream analysis.

Sure! Here are the tables in a copiable format:
generate_pv arguments:
ArgumentShorthandDescriptioncsv_file-cThe path to the CSV file containing the spatial transcriptomics data.img_file_path-iThe path to the image file associated with the spatial data (optional).hexagon_size-hsThe size of the hexagons used for binning (default: 100).output_path-oThe path to save the Pseudovisium output (default: current directory).batch_size-bThe number of rows per batch for parallel processing (default: 1000000).alignment_matrix_file-amThe path to the alignment matrix file (optional).project_name-pThe name of the project (default: 'project').image_pixels_per_um-ppuThe number of image pixels per micrometer (default: 1).tissue_hires_scalef-thsThe scaling factor for the high-resolution tissue image (default: 0.2).technology-tThe technology used for the spatial data (default: 'Xenium').feature_colname-fcThe name of the feature column in the CSV file (default: 'feature_name').x_colname-xcThe name of the x-coordinate column in the CSV file (default: 'x_location').y_colname-ycThe name of the y-coordinate column in the CSV file (default: 'y_location').cell_id_colname-ccThe name of the cell ID column in the CSV file (default: 'None').quality_colname-qcolThe name of the quality score column in the CSV file (default: 'qv').pixel_to_micron-ptmWhether to convert pixel coordinates to micron coordinates (default: False).max_workers--mwThe maximum number of worker processes to use for parallel processing (default: min(2, multiprocessing.cpu_count())).quality_filter-qfWhether to filter rows based on quality score (default: False).count_colname-ccolThe name of the count column in the CSV file (default: 'NA').visium_hd_folder-vhfThe path to the Visium HD folder (optional).smoothing-sThe smoothing factor for high-resolution data (default: False).quality_per_hexagon-qphWhether to calculate quality per hexagon (default: False).quality_per_probe-qppWhether to calculate quality per probe (default: False).h5_x_colname-h5xThe name of the x-coordinate column in the h5 file (default: 'x').h5_y_colname-h5yThe name of the y-coordinate column in the h5 file (default: 'y').move_x-mxThe amount to move the x-coordinate (default: 0).move_y-myThe amount to move the y-coordinate (default: 0).coord_to_um_conversion-ctuThe conversion factor from coordinates to micrometers (default: 1).spot_diameter-sdThe diameter of the spot for Visium-like array structure (optional).
qc arguments:

### 2. qc:   Quality Control 📈

Pseudovisium's qc command generates a detailed quality control (QC) report for a set of Pseudovisium/Visium formatted replicates. It calculates a wide range of metrics, including the number of hexagons with a minimum count threshold, the number of genes present in a certain percentage of hexagons, and the median counts and features per hexagon. The report also showcases hexagon plots for selected genes and provides comparison plots between different datasets, allowing you to assess the quality and consistency of your data.

### 3. merge:   Data Merging 🧩

With the merge command, you can easily merge multiple Pseudovisium or Visium format files. This feature allows you to combine data from different datasets, merge images together, and generate a merged output directory in the Pseudovisium/Visium format. Pseudovisium makes it effortless to merge data from multiple spatial transcriptomics experiments, enabling comprehensive analysis across datasets.


## 🎯 Flexibility and Compatibility

Pseudovisium is designed to be flexible and compatible with various spatial transcriptomics technologies. It supports data from different platforms ensuring seamless integration with other analysis tools and workflows. Input files range from transcripts.csv, to .h5 as well as 10X feature-barcode-matrix directories.
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

Converting 10X Visium HD Mouse brain data to Pseudovisium format:

https://github.com/BKover99/Pseudovisium/blob/main/Visium_HD_example.ipynb

QC-ing 28 replicates of Xenium pulmonary lung dataset:

https://github.com/BKover99/Pseudovisium/blob/main/Pulmonary_lung_QC.ipynb

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
