# <span style="font-size:70px; color:#FF4500;">⬢</span> <span style="font-size:70px; color:#FF4500;">⬢</span>  Pseudovisium <span style="font-size:70px; color:#FF4500;">⬢</span> <span style="font-size:70px; color:#FF4500;">⬢</span> 
Pseudovisium is a Python software package designed to democratize the analysis of spatial transcriptomics data. By leveraging hexagonal binning, Pseudovisium enables efficient compression and visualization of spatial data, making exploratory analysis and quality control at least an order of magnitude faster and more memory efficient. The goal of this tool is not to increase accuracy, but to make spatial data analysis more accessible, regardless of computing environment. Additionally, this package facilitates simulating low-res/Visium spatial experiments both for practical (e.g. experimental design) and theoretical (e.g. studying the impact of resolution) purposes.

## 🚀 Key Features
### 1. *pseudovisium_generate* module: Data Compression 📊
The generate_pv command takes your raw spatial transcriptomics data in CSV/parquet format, or your spatial transcriptomics object in AnnData/SpatialData/Zarr format, and performs spatial binning (hexagonal or square) to compress the data while preserving spatial information. See the [extensive list of tutorials](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/) for possible use cases. It calculates binned counts and cell counts, and creates a well-structured output directory with all the necessary files for downstream analysis. The output format is the Pseudovisium format, which can then be used as any Visium folder (for any downstream application) or as input to pseudovisium_qc and/or pseudovisium_merge.

#### CLI arguments for *pseudovisium_generate* or generate_pv() (e.g. used within Jupyter notebook) function:
| Argument | Shorthand | Description |
| --- | --- | --- |
| `csv_file` | `-c` | Path to input data file (CSV, Parquet, or gzipped CSV). |
| `img_file_path` | `-i` | Path to tissue image file (optional). |
| `bin_size` | `-bs` | Size of the spatial bins (default: 100). |
| `output_path` | `-o` | Base path for output directory (default: current directory). |
| `batch_size` | `-b` | Number of rows per batch for parallel processing (default: 1000000). |
| `alignment_matrix_file` | `-am` | Path to image alignment matrix (optional). |
| `project_name` | `-p` | Name of the project subfolder (default: 'project'). |
| `image_pixels_per_um` | `-ppu` | Image resolution in pixels per micrometer (default: 1). |
| `tissue_hires_scalef` | `-ths` | Scaling factor for high-resolution tissue image (default: 0.2). |
| `technology` | `-t` | Technology platform name (default: 'Xenium'). Supports: Xenium, Vizgen, CosMx/Nanostring, Visium_HD, Curio, seqFISH, Zarr, SpatialData, AnnData. |
| `feature_colname` | `-fc` | Name of the feature/gene column (default: 'feature_name'). |
| `x_colname` | `-xc` | Name of the x-coordinate column (default: 'x_location'). |
| `y_colname` | `-yc` | Name of the y-coordinate column (default: 'y_location'). |
| `cell_id_colname` | `-cc` | Name of the cell ID column (default: 'None'). |
| `quality_colname` | `-qcol` | Name of the quality score column (default: 'qv'). |
| `max_workers` | `--mw` | Maximum number of parallel processes (default: min(2, multiprocessing.cpu_count())). |
| `quality_filter` | `-qf` | Whether to filter rows based on quality score (default: False). |
| `count_colname` | `-ccol` | Name of the count column (default: 'NA'). |
| `folder_or_object` | `-foo` | Path to data folder or data object for specific formats (optional). |
| `smoothing` | `-s` | Smoothing factor for high-resolution data (default: False). |
| `quality_per_hexagon` | `-qph` | Whether to calculate quality metrics per bin (default: False). |
| `quality_per_probe` | `-qpp` | Whether to calculate quality metrics per probe (default: False). |
| `h5_x_colname` | `-h5x` | Name of x-coordinate column in h5 files (default: 'x'). |
| `h5_y_colname` | `-h5y` | Name of y-coordinate column in h5 files (default: 'y'). |
| `coord_to_um_conversion` | `-ctu` | Conversion factor from coordinates to micrometers (default: 1). |
| `spot_diameter` | `-sd` | Diameter for Visium-like spot array simulation (optional). |
| `hex_square` | `-hex` | Shape of spatial bins: "hex" or "square" (default: "hex"). |
| `sd_table_id` | `-sid` | Table identifier in SpatialData object (optional). |
| `pixel_to_micron` | `-ptm` | Whether to convert pixel coordinates to microns (default: False). |


### 2. *pseudovisium_qc* module:  Quality Control 📈

Pseudovisium's qc command generates a detailed quality control (QC) report for a set of Pseudovisium/Visium formatted replicates. It calculates a wide range of metrics, including the number of hexagons with a minimum count threshold, the number of genes present in a certain percentage of hexagons, and the median counts and features per hexagon. The report also showcases hexagon plots for selected genes and provides comparison plots between different datasets, allowing you to assess the quality and consistency of your data.

#### CLI arguments for *pseudovisium_qc* or generate_qc_report() (e.g. used within Jupyter notebook) function:

| Argument | Shorthand | Description |
| --- | --- | --- |
| `folders` | `-f` | List of folders containing Pseudovisium/Visium output |
| `output_folder` | `-o` | Output folder path (default: current working directory). |
| `gene_names` | `-g` | List of gene names to plot (default: ["RYR3", "AQP4", "THBS1"]). |
| `include_morans_i` | `-m` | Include Moran's I features tab (default: False). |
| `max_workers` | `--mw` | Number of workers to use for parallel processing (default: 4). |
| `normalisation` | `-n` | Normalise the counts by the total counts per cell (default: False). |
| `save_plots` | `-sp` | Save generated plots as publication ready figures (default: False). |
| `squidpy` | `-sq` | Use squidpy to calculate Moran's I (default: False). |
| `minimal_plots` | `-mp` | Generate minimal plots by excluding heatmaps and individual comparison plots (default: False). |
| `neg_ctrl_string` | `-nc` | String to identify negative control probes (default: "control\|ctrl\|code\|Code\|assign\|Assign\|pos\|NegPrb\|neg\|Ctrl\|blank\|Control\|Blank\|BLANK"). |


### 3. *pseudovisium_merge* module:     Data Merging 🧩 

With the merge command, you can easily merge multiple Pseudovisium or Visium format files. This feature allows you to combine data from different datasets, merge images together, and generate a merged output directory in the Pseudovisium/Visium format. Pseudovisium_merge makes it effortless to merge data from multiple spatial transcriptomics experiments, enabling comprehensive analysis across datasets.

#### CLI arguments for *pseudovisium_merge* or merge_visium() (e.g. used within Jupyter notebook) function:

| Argument | Shorthand | Description |
| --- | --- | --- |
| `folders` | `-f` | List of folders containing Pseudovisium/Visium output |
| `output_path` | `-o` | Output folder path (default: current working directory). |
| `project_name` | `-p` | Project name for output (default: visium_merged). |
| `pv_format` | `-pvf` | Indicate if input is in Pseudovisium format (default: False).|
| `only_common` | `--oc` | Only keep genes present in all datasets (default: False). |



## 🎯 Flexibility and Compatibility

Pseudovisium is designed to be flexible and compatible with various spatial transcriptomics technologies. It supports data from different platforms ensuring seamless integration with other analysis tools and workflows. Input files range from transcripts.csv, transcripts.parquet to .h5, .h5ad as well as 10X feature-barcode-matrix directories.
Technologies tried include:

#### MERSCOPE (Vizgen)
#### Xenium (10X)
#### CosMx (Nanostring)
#### Slide-seq (Curio Seeker)
#### seqFISH (Spatial Genomics)
#### VisiumHD (10X)

In addition we also support various inputs from the scverse eco-system, including:
#### AnnData
#### SpatialData
#### Zarr


Tested operating systems include:
#### Windows 10
#### MacOS Sonoma 14.3.1

## 🚀 Get Started with Pseudovisium

To start using Pseudovisium, simply install the package and explore the documentation and examples provided. Pseudovisium is open-source and actively maintained, ensuring continuous improvements and support for the spatial transcriptomics community.

Pseudovisium is available on PyPI and can be easily installed using pip:

```python
pip install Pseudovisium
```

For more information and the latest version, visit the Pseudovisium PyPI page https://pypi.org/project/Pseudovisium/.

## Compatibility with AnnData / scanpy / squidpy framework - *adata_to_adata*
Initially the purpose of Pseudovisium was to operate entirely on raw files - raw high-res input to raw PV output. But due to community demands, we now also enable binning of *cells* (usually cells, but works for transcripts if each observation in your AnnData is a transcript) directly from AnnData objects. This is extremely fast (few seconds usually), and might be convenient for a variety of downstream applications.

It is a very straightforward single line of code.
```python
import Pseudovisium.pseudovisium_generate as pvg
adata_new = pvg.adata_to_adata(adata_fullres,25,"hex")
```
## Compatibility with SpatialData framework - *spatialdata_to_spatialdata*





[Converting Nanostring CosMx Pancreas AnnData object to binned data.](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/Working_on_anndata_Tutorial.ipynb)

## Use cases for binning/rasterisation

#### Speeding up analysis and QC (https://doi.org/10.1101/2024.07.23.604776)
#### Facilitating spatial alignment and 3D atlas building (More to come...)

## Examples
[See the example Google Colab on converting 10X Xenium Mouse pup data to Pseudovisium format](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/Xenium_Tutorial.ipynb)

[Converting 10X Visium HD Mouse brain data to Pseudovisium format](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/Visium_HD_Tutorial.ipynb)

[Converting Nanostring CosMx Pancreas data to Pseudovisium](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/CosMx_Tutorial.ipynb)

[Converting Slide-seq brain data to Pseudovisium](https://github.com/BKover99/Pseudovisium/blob/main/Tutorials/Slideseq_Tutorial.ipynb)

[Reproducing all the analysis from the Pseudovisium pre-print](https://github.com/BKover99/Pseudovisium/tree/main/Paper%20figures)

[Performing spatial GSEA with spatialAUC](https://github.com/BKover99/spatialAUC)

## Goals tracker
- [x] Starting rotation project with working prototype - Apr 7, 2024
- [x] Releasing the repo with minimal working capabilities - Apr 20, 2024
- [x] Adding supporting Colab notebooks - Jun 20, 2024
- [x] Releasing all data used in this work on OneDrive - Jun 20, 2024
- [x] Tidying up all code - Jun 20, 2024
- [x] Posting pre-print to Biorxiv - July 24, 2024  https://www.biorxiv.org/content/10.1101/2024.07.23.604776v1.full
- [x] Peer-review
- [ ] Revisions - currently ongoing, expected to be done soon.
- [ ] Publication of peer-reviewed research article

## Citation
To cite Pseudovisium, cite the Pseudovisium pre-print.
https://www.biorxiv.org/content/10.1101/2024.07.23.604776v1

DOI: https://doi.org/10.1101/2024.07.23.604776

*Kövér B, Vigilante A. Rapid and memory-efficient analysis and quality control of large spatial transcriptomics datasets [Internet]. bioRxiv; 2024. p. 2024.07.23.604776. Available from: https://www.biorxiv.org/content/10.1101/2024.07.23.604776v1*




### Author
Bence Kövér
https://twitter.com/kover_bence 
https://www.linkedin.com/in/ben-kover/
