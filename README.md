# 🛑 Pseudovisium
Pseudovisium is a Python software package designed to democratize the analysis of spatial transcriptomics data. By leveraging hexagonal binning, Pseudovisium enables efficient compression and visualization of spatial data, making exploratory analysis and quality control at least an order of magnitude faster and more memory efficient. The goal of this tool is not to increase accuracy, but to make spatial data analysis more accessible, regardless of computing environment. Additionally, this package facilitates simulating low-res/Visium spatial experiments both for practical (e.g. experimental design) and theoretical (e.g. studying the impact of resolution) purposes.

## 🚀 Key Features
### 1. generate_pv:   Data Compression 📊

The generate_pv command takes your spatial transcriptomics data in CSV format and performs hexagonal binning to compress the data while preserving spatial information. It calculates hexagon counts and cell counts, and creates a well-structured output directory with all the necessary files for downstream analysis.

#### generate_pv arguments:

| Argument | Shorthand | Description |
| --- | --- | --- |
| `csv_file` | `-c` | The path to the CSV file containing the spatial transcriptomics data. |
| `img_file_path` | `-i` | The path to the image file associated with the spatial data (optional). |
| `hexagon_size` | `-hs` | The size of the hexagons used for binning (default: 100). |
| `output_path` | `-o` | The path to save the Pseudovisium output (default: current directory). |
| `batch_size` | `-b` | The number of rows per batch for parallel processing (default: 1000000). |
| `alignment_matrix_file` | `-am` | The path to the alignment matrix file (optional). |
| `project_name` | `-p` | The name of the project (default: 'project'). |
| `image_pixels_per_um` | `-ppu` | The number of image pixels per micrometer (default: 1). |
| `tissue_hires_scalef` | `-ths` | The scaling factor for the high-resolution tissue image (default: 0.2). |
| `technology` | `-t` | The technology used for the spatial data (default: 'Xenium'). |
| `feature_colname` | `-fc` | The name of the feature column in the CSV file (default: 'feature_name'). |
| `x_colname` | `-xc` | The name of the x-coordinate column in the CSV file (default: 'x_location'). |
| `y_colname` | `-yc` | The name of the y-coordinate column in the CSV file (default: 'y_location'). |
| `cell_id_colname` | `-cc` | The name of the cell ID column in the CSV file (default: 'None'). |
| `quality_colname` | `-qcol` | The name of the quality score column in the CSV file (default: 'qv'). |
| `max_workers` | `--mw` | The maximum number of worker processes to use for parallel processing (default: min(2, multiprocessing.cpu_count())). |
| `quality_filter` | `-qf` | Whether to filter rows based on quality score (default: False). |
| `count_colname` | `-ccol` | The name of the count column in the CSV file (default: 'NA'). |
| `visium_hd_folder` | `-vhf` | The path to the Visium HD folder (optional). |
| `smoothing` | `-s` | The smoothing factor for high-resolution data (default: False). |
| `quality_per_hexagon` | `-qph` | Whether to calculate quality per hexagon (default: False). |
| `quality_per_probe` | `-qpp` | Whether to calculate quality per probe (default: False). |
| `h5_x_colname` | `-h5x` | The name of the x-coordinate column in the h5 file (default: 'x'). |
| `h5_y_colname` | `-h5y` | The name of the y-coordinate column in the h5 file (default: 'y'). |
| `coord_to_um_conversion` | `-ctu` | The conversion factor from coordinates to micrometers (default: 1). |
| `spot_diameter` | `-sd` | The diameter of the spot for Visium-like array structure (optional). |


### 2. qc:   Quality Control 📈

Pseudovisium's qc command generates a detailed quality control (QC) report for a set of Pseudovisium/Visium formatted replicates. It calculates a wide range of metrics, including the number of hexagons with a minimum count threshold, the number of genes present in a certain percentage of hexagons, and the median counts and features per hexagon. The report also showcases hexagon plots for selected genes and provides comparison plots between different datasets, allowing you to assess the quality and consistency of your data.

#### qc arguments:

| Argument | Shorthand | Description |
| --- | --- | --- |
| `folders` | `-f` | List of folders containing Pseudovisium output. |
| `output_folder` | `-o` | Output folder path (default: current working directory). |
| `gene_names` | `-g` | List of gene names to plot (default: ["RYR3", "AQP4", "THBS1"]). |
| `include_morans_i` | `-m` | Include Moran's I features tab (default: False). |
| `max_workers` | `--mw` | Number of workers to use for parallel processing (default: 4). |
| `normalisation` | `-n` | Normalise the counts by the total counts per cell (default: False). |
| `save_plots` | `-sp` | Save generated plots as publication ready figures (default: False). |


### 3. merge:   Data Merging 🧩 - Not finalised

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
