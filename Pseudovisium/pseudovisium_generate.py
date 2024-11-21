import csv
import numpy as np
import pandas as pd
import cv2
import json
import gzip
import concurrent.futures
import os
import shutil
from tqdm import tqdm
import itertools
import argparse
import tifffile
import multiprocessing
import time
import pyarrow.parquet as pq
import scanpy as sc
import scipy.io
import h5py
import scipy.sparse
from pathlib import Path
import subprocess
import datetime
from numba import jit
import spatialdata as sd
import anndata as ad


@jit(nopython=True)
def closest_hex(x, y, bin_size, spot_diameter=None):
    """
    Calculates the closest hexagon centroid to the given (x, y) coordinates.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        bin_size (float): The size of the hexagon (distance from center to middle of edge).
        spot_diameter (float, optional): The diameter of the spot for Visium-like array simulation.
                                       If provided, points too far from centroids are excluded.

    Returns:
        tuple or str: The closest hexagon centroid coordinates (x, y) rounded to the nearest integer.
                     Returns "-1" as a string if spot_diameter is provided and the distance to
                     the closest hexagon centroid is greater than half the spot diameter.
    """
    spot = spot_diameter is not None

    x_ = x // (bin_size * 2)
    y_ = y // (bin_size * 1.732050807)

    if y_ % 2 == 1:
        # lower_left
        option_1_hexagon = (x_ * 2 * bin_size, y_ * 1.732050807 * bin_size)

        # lower right
        option_2_hexagon = (
            (x_ + 1) * 2 * bin_size,
            y_ * 1.732050807 * bin_size,
        )

        # upper middle
        option_3_hexagon = (
            (x_ + 0.5) * 2 * bin_size,
            (y_ + 1) * 1.732050807 * bin_size,
        )

    else:
        # lower middle
        option_1_hexagon = (
            (x_ + 0.5) * 2 * bin_size,
            y_ * (1.732050807 * bin_size),
        )

        # upper left
        option_2_hexagon = (
            x_ * 2 * bin_size,
            (y_ + 1) * 1.732050807 * bin_size,
        )

        # upper right
        option_3_hexagon = (
            (x_ + 1) * 2 * bin_size,
            (y_ + 1) * 1.732050807 * bin_size,
        )

    # Calculate distances
    distance_1 = np.sqrt(
        (x - option_1_hexagon[0]) ** 2 + (y - option_1_hexagon[1]) ** 2
    )
    distance_2 = np.sqrt(
        (x - option_2_hexagon[0]) ** 2 + (y - option_2_hexagon[1]) ** 2
    )
    distance_3 = np.sqrt(
        (x - option_3_hexagon[0]) ** 2 + (y - option_3_hexagon[1]) ** 2
    )

    # Find the minimum distance
    min_distance = min(distance_1, distance_2, distance_3)

    # Select the closest hexagon
    if min_distance == distance_1:
        closest = option_1_hexagon
    elif min_distance == distance_2:
        closest = option_2_hexagon
    else:
        closest = option_3_hexagon

    closest = (round(closest[0], 0), round(closest[1], 1))

    if spot:
        if np.sqrt((x - closest[0]) ** 2 + (y - closest[1]) ** 2) < spot_diameter / 2:
            return closest
        else:
            return str(-1)
    else:
        return closest


def closest_square(x, y, square_size):
    """
    Calculates the closest square centroid to the given (x, y) coordinates.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        square_size (float): The length of the square side.

    Returns:
        tuple: The closest square centroid coordinates (x, y) rounded to the nearest integer.
    """

    x_ = x // square_size
    y_ = y // square_size
    return (round((x_ + 0.5) * square_size, 0), round((y_ + 0.5) * square_size, 0))


def process_batch(
    df_batch,
    bin_size,
    feature_colname,
    x_colname,
    y_colname,
    cell_id_colname,
    quality_colname=None,
    quality_filter=False,
    count_colname="NA",
    smoothing=False,
    quality_per_hexagon=False,
    quality_per_probe=False,
    coord_to_um_conversion=1,
    spot_diameter=None,
    hex_square="hex",
):
    """
    Processes a batch of spatial transcriptomics data to calculate binned counts and statistics.

    Args:
        df_batch (pd.DataFrame): DataFrame containing the batch data.
        bin_size (float): Size of the spatial bin (hexagon or square).
        feature_colname (str): Name of the feature/gene column.
        x_colname (str): Name of the x-coordinate column.
        y_colname (str): Name of the y-coordinate column.
        cell_id_colname (str): Name of the cell ID column.
        quality_colname (str, optional): Name of the quality score column.
        quality_filter (bool, optional): Whether to filter rows based on quality score.
        count_colname (str, optional): Name of the count column. Uses 1 if "NA".
        smoothing (bool, optional): Whether to apply smoothing to the counts.
        quality_per_hexagon (bool, optional): Whether to calculate quality metrics per spatial bin.
        quality_per_probe (bool, optional): Whether to calculate quality metrics per probe/gene.
        coord_to_um_conversion (float, optional): Conversion factor from coordinates to micrometers.
        spot_diameter (float, optional): Diameter for Visium-like spot array simulation.
        hex_square (str, optional): Shape of spatial bins: "hex" or "square".

    Returns:
        tuple: Contains combination of:
            - DataFrame of binned counts
            - DataFrame of binned cell counts (if cell_id_colname provided)
            - Dictionary of bin quality metrics (if quality_per_hexagon True)
            - Dictionary of probe quality metrics (if quality_per_probe True)
    """

    df_batch[x_colname] = pd.to_numeric(df_batch[x_colname], errors="coerce")
    df_batch[y_colname] = pd.to_numeric(df_batch[y_colname], errors="coerce")

    # adjusting coordinates
    df_batch[x_colname] = (df_batch[x_colname]) * coord_to_um_conversion
    df_batch[y_colname] = (df_batch[y_colname]) * coord_to_um_conversion

    # smoothing, generally only for Visium HD or Curio
    if smoothing != False:
        for i, j in zip(
            [smoothing, smoothing, -smoothing, -smoothing],
            [smoothing, -smoothing, smoothing, -smoothing],
        ):
            df_batch_changed = df_batch.copy()
            df_batch_changed[x_colname] = df_batch_changed[x_colname] + i
            df_batch_changed[y_colname] = df_batch_changed[y_colname] + j
            df_batch = pd.concat([df_batch, df_batch_changed])
        # reset_index
        df_batch.reset_index(drop=True, inplace=True)
        # remove first df_batch_changed length rows
        df_batch = df_batch.iloc[df_batch_changed.shape[0] :]

    # based on x and y column, apply closest_hex. If spot_diameter is set, then use that to recapitulate raw visium
    if hex_square == "hex":
        hexagons = np.array(
            [
                str(closest_hex(x, y, bin_size, spot_diameter))
                for x, y in zip(df_batch[x_colname], df_batch[y_colname])
            ]
        )
    elif hex_square == "square":
        hexagons = np.array(
            [
                str(closest_square(x, y, bin_size))
                for x, y in zip(df_batch[x_colname], df_batch[y_colname])
            ]
        )
    df_batch["hexagons"] = hexagons
    # filter out rows where hexagon is -1
    df_batch = df_batch[df_batch["hexagons"] != "-1"]

    # create a dok matrix to store the counts, which is
    counts = (
        np.ones(df_batch.shape[0])
        if count_colname == "NA"
        else df_batch[count_colname].values
    )
    if smoothing != False:
        counts = counts / 4

    df_batch["counts"] = counts

    if quality_filter or quality_per_hexagon or quality_per_probe:
        df_batch[quality_colname] = pd.to_numeric(
            df_batch[quality_colname], errors="coerce"
        )

    if quality_per_hexagon == True:
        hexagon_quality = df_batch.groupby("hexagons")[quality_colname].agg(
            ["mean", "count"]
        )
        if isinstance(hexagon_quality, pd.DataFrame):
            hexagon_quality = hexagon_quality.to_dict(orient="index")
        else:
            print(
                "hexagon_quality is not a DataFrame. Skipping conversion to dictionary.\n"
            )

    if quality_per_probe == True:
        # create probe_quality from df_batch
        probe_quality = df_batch.groupby(feature_colname)[quality_colname].agg(
            ["mean", "count"]
        )
        if isinstance(probe_quality, pd.DataFrame):
            probe_quality = probe_quality.to_dict(orient="index")
        else:
            print(
                "probe_quality is not a DataFrame. Skipping conversion to dictionary.\n"
            )

    if quality_filter:
        df_batch = df_batch[df_batch[quality_colname] > 20]

    hexagon_counts = (
        df_batch[["hexagons", feature_colname, "counts"]]
        .groupby(["hexagons", feature_colname])
        .aggregate({"counts": "sum"})
        .reset_index()
    )

    returning_items = [hexagon_counts]

    if cell_id_colname != "None":
        hexagon_cell_counts = (
            df_batch[["hexagons", cell_id_colname, "counts"]]
            .groupby(["hexagons", cell_id_colname])
            .aggregate({"counts": "sum"})
            .reset_index()
        )

        returning_items.append(hexagon_cell_counts)

    if quality_per_hexagon == True:
        returning_items.append(hexagon_quality)
    if quality_per_probe == True:
        returning_items.append(probe_quality)

    return tuple(returning_items)


def write_10X_h5(adata, file):
    """
    Writes an AnnData object to a 10X-formatted h5 file.

    Args:
        adata (AnnData): AnnData object to be written.
        file (str): Output file path. '.h5' extension added if not provided.

    Notes:
        Creates a 10X Genomics-compatible h5 file
    """

    if ".h5" not in file:
        file = f"{file}.h5"
    if Path(file).exists():
        print(f"File `{file}` already exists. Removing it.\n")
        os.remove(file)

    adata.var["feature_types"] = ["Gene Expression" for _ in range(adata.var.shape[0])]
    adata.var["genome"] = ["pv_placeholder" for _ in range(adata.var.shape[0])]
    adata.var["gene_ids"] = adata.var.index

    w = h5py.File(file, "w")
    grp = w.create_group("matrix")
    grp.create_dataset("barcodes", data=adata.obs_names.values.astype("|S"))

    X = adata.X.T.tocsc()  # Convert the matrix to CSC format
    grp.create_dataset("data", data=X.data.astype(np.float32))
    grp.create_dataset("indices", data=X.indices.astype(np.int32))
    grp.create_dataset("indptr", data=X.indptr.astype(np.int32))
    grp.create_dataset("shape", data=np.array(X.shape).astype(np.int32))

    ftrs = grp.create_group("features")
    if "feature_types" in adata.var:
        ftrs.create_dataset(
            "feature_type", data=adata.var.feature_types.values.astype("|S")
        )
    if "genome" in adata.var:
        ftrs.create_dataset("genome", data=adata.var.genome.values.astype("|S"))
    if "gene_ids" in adata.var:
        ftrs.create_dataset("id", data=adata.var.gene_ids.values.astype("|S"))
    ftrs.create_dataset("name", data=adata.var.index.values.astype("|S"))

    w.close()


def process_csv_file(
    csv_file,
    bin_size,
    batch_size=1000000,
    feature_colname="feature_name",
    x_colname="x_location",
    y_colname="y_location",
    cell_id_colname="None",
    quality_colname="qv",
    max_workers=min(2, multiprocessing.cpu_count()),
    quality_filter=False,
    count_colname="NA",
    smoothing=False,
    quality_per_hexagon=False,
    quality_per_probe=False,
    coord_to_um_conversion=1,
    spot_diameter=None,
    hex_square="hex",
):
    """
    Processes spatial transcriptomics data to calculate binned counts using parallel processing.

    Args:
        csv_file (str): Path to CSV, Parquet, or gzipped CSV file containing spatial data.
        bin_size (float): Size of the spatial bin (hexagon or square).
        batch_size (int, optional): Number of rows to process per batch.
        feature_colname (str, optional): Name of the feature/gene column.
        x_colname (str, optional): Name of the x-coordinate column.
        y_colname (str, optional): Name of the y-coordinate column.
        cell_id_colname (str, optional): Name of the cell ID column.
        quality_colname (str, optional): Name of the quality score column.
        max_workers (int, optional): Maximum number of parallel processes.
        quality_filter (bool, optional): Whether to filter rows based on quality score.
        count_colname (str, optional): Name of the count column. Uses 1 if "NA".
        smoothing (bool, optional): Whether to apply smoothing to the counts.
        quality_per_hexagon (bool, optional): Whether to calculate quality metrics per spatial bin.
        quality_per_probe (bool, optional): Whether to calculate quality metrics per probe/gene.
        coord_to_um_conversion (float, optional): Conversion factor from coordinates to micrometers.
        spot_diameter (float, optional): Diameter for Visium-like spot array simulation.
        hex_square (str, optional): Shape of spatial bins: "hex" or "square".

    Returns:
        tuple: Contains:
            - scipy.sparse.csr_matrix: Binned count matrix
            - np.ndarray: Unique bin coordinates
            - np.ndarray: Unique features/genes
            - pd.DataFrame: Binned cell counts (if cell_id_colname provided)
            - dict: Bin quality metrics (if quality_per_hexagon True)
            - dict: Probe quality metrics (if quality_per_probe True)
    """

    print(f"Quality filter is set to {quality_filter}\n")
    print(f"Quality counting per hexagon is set to {quality_per_hexagon}\n")
    print(f"Quality counting per probe is set to {quality_per_probe}\n")
    spot = True if spot_diameter != None else False
    if spot:
        print(
            "Visium-like spots are going to be used rather than hexagonal tesselation!!!\n"
        )

    fieldnames = [feature_colname, x_colname, y_colname]
    if cell_id_colname != "None":
        fieldnames.append(cell_id_colname)
    if (
        quality_filter == True
        or quality_per_hexagon == True
        or quality_per_probe == True
    ):
        fieldnames.append(quality_colname)
    if count_colname != "NA":
        fieldnames.append(count_colname)

    hexagon_quality = {}
    probe_quality = {}
    hexagon_counts = pd.DataFrame()
    hexagon_cell_counts = pd.DataFrame()
    n_process = min(max_workers, multiprocessing.cpu_count())
    print(f"Processing batches using {n_process} processes\n")

    is_parquet = csv_file.endswith(".parquet")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if is_parquet:
            print("Reading parquet file...\n")
            parquet_file = pq.ParquetFile(csv_file)
            num_batches = -(
                -parquet_file.metadata.num_rows // batch_size
            )  # Ceiling division
            futures = [
                executor.submit(
                    process_batch,
                    batch.to_pandas()[fieldnames],
                    bin_size,
                    feature_colname,
                    x_colname,
                    y_colname,
                    cell_id_colname,
                    quality_colname,
                    quality_filter,
                    count_colname,
                    smoothing,
                    quality_per_hexagon,
                    quality_per_probe,
                    coord_to_um_conversion,
                    spot_diameter,
                    hex_square,
                )
                for batch in parquet_file.iter_batches(batch_size=batch_size)
            ]
        else:
            file_open_fn = gzip.open if csv_file.endswith(".gz") else open
            file_open_mode = "rt" if csv_file.endswith(".gz") else "r"
            with file_open_fn(csv_file, file_open_mode) as file:
                reader = csv.DictReader(file)
                futures = []
                while True:
                    chunk = list(itertools.islice(reader, batch_size))
                    if not chunk:
                        break
                    df_chunk = pd.DataFrame(chunk, columns=fieldnames)
                    futures.append(
                        executor.submit(
                            process_batch,
                            df_chunk,
                            bin_size,
                            feature_colname,
                            x_colname,
                            y_colname,
                            cell_id_colname,
                            quality_colname,
                            quality_filter,
                            count_colname,
                            smoothing,
                            quality_per_hexagon,
                            quality_per_probe,
                            coord_to_um_conversion,
                            spot_diameter,
                            hex_square,
                        )
                    )

        with tqdm(
            total=len(futures), desc="Processing batches", unit="batch"
        ) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                all_res = future.result()
                batch_hexagon_counts = all_res[0]
                # stack the batch_hexagon_counts under the hexagon_counts
                hexagon_counts = pd.concat(
                    [hexagon_counts, batch_hexagon_counts], axis=0
                )

                hexagon_counts = (
                    hexagon_counts[["hexagons", feature_colname, "counts"]]
                    .groupby(["hexagons", feature_colname])
                    .aggregate({"counts": "sum"})
                    .reset_index()
                )

                if cell_id_colname != "None":
                    batch_hexagon_cell_counts = all_res[1]

                    hexagon_cell_counts = pd.concat(
                        [hexagon_cell_counts, batch_hexagon_cell_counts], axis=0
                    )

                    hexagon_cell_counts = (
                        hexagon_cell_counts[["hexagons", cell_id_colname, "counts"]]
                        .groupby(["hexagons", cell_id_colname])
                        .aggregate({"counts": "sum"})
                        .reset_index()
                    )

                if quality_per_hexagon == True:
                    if cell_id_colname != "None":
                        batch_hexagon_quality = all_res[2]
                    else:
                        batch_hexagon_quality = all_res[1]
                    for (
                        hexagon_quality_hex,
                        quality_dict,
                    ) in batch_hexagon_quality.items():
                        try:
                            if hexagon_quality_hex not in hexagon_quality:
                                hexagon_quality[hexagon_quality_hex] = quality_dict
                            else:
                                hexagon_quality[hexagon_quality_hex]["mean"] = (
                                    float(hexagon_quality[hexagon_quality_hex]["mean"])
                                    * hexagon_quality[hexagon_quality_hex]["count"]
                                    + float(quality_dict["mean"])
                                    * quality_dict["count"]
                                ) / (
                                    hexagon_quality[hexagon_quality_hex]["count"]
                                    + quality_dict["count"]
                                )
                                hexagon_quality[hexagon_quality_hex][
                                    "count"
                                ] += quality_dict["count"]
                        except KeyError:
                            print(f"Error in trying to add to hexagon_quality\n")

                if quality_per_probe == True:
                    if cell_id_colname != "None" and quality_per_hexagon == True:
                        batch_probe_quality = all_res[3]
                    elif cell_id_colname != "None" and quality_per_hexagon == False:
                        batch_probe_quality = all_res[2]
                    elif cell_id_colname == "None" and quality_per_hexagon == True:
                        batch_probe_quality = all_res[2]
                    else:
                        batch_probe_quality = all_res[1]

                    for probe, quality_dict in batch_probe_quality.items():
                        try:
                            if probe not in probe_quality:
                                probe_quality[probe] = quality_dict
                            else:
                                probe_quality[probe]["mean"] = (
                                    float(probe_quality[probe]["mean"])
                                    * probe_quality[probe]["count"]
                                    + float(quality_dict["mean"])
                                    * quality_dict["count"]
                                ) / (
                                    probe_quality[probe]["count"]
                                    + quality_dict["count"]
                                )
                                probe_quality[probe]["count"] += quality_dict["count"]
                        except KeyError:
                            print(f"Error in trying to add to probe_quality\n")
                progress_bar.update(1)

    unique_hexagons = hexagon_counts["hexagons"].unique()
    hexagon_counts["hexagon_id"] = hexagon_counts["hexagons"].map(
        {hexagon: i for i, hexagon in enumerate(unique_hexagons)}
    )
    if cell_id_colname != "None":
        hexagon_cell_counts["hexagon_id"] = hexagon_cell_counts["hexagons"].map(
            {hexagon: i for i, hexagon in enumerate(unique_hexagons)}
        )

    unique_features = hexagon_counts[feature_colname].unique()
    # turn to np array and sort
    unique_features = np.sort(unique_features)

    hexagon_counts[feature_colname] = hexagon_counts[feature_colname].map(
        {feature: i for i, feature in enumerate(unique_features)}
    )

    hexagon_counts = scipy.sparse.csr_matrix(
        (
            hexagon_counts["counts"],
            (hexagon_counts["hexagon_id"], hexagon_counts[feature_colname]),
        ),
        shape=(len(hexagon_counts["hexagons"].unique()), len(unique_features)),
    )

    return (
        hexagon_counts,
        unique_hexagons,
        unique_features,
        hexagon_cell_counts,
        hexagon_quality,
        probe_quality,
    )


def create_pseudovisium(
    output_path,
    hexagon_counts,
    unique_hexagons,
    unique_features,
    hexagon_cell_counts,
    hexagon_quality,
    probe_quality,
    cell_id_colname,
    img_file_path=None,
    shift_to_positive=False,
    project_name="project",
    alignment_matrix_file=None,
    image_pixels_per_um=1.0,
    bin_size=100,
    tissue_hires_scalef=0.2,
    pixel_to_micron=False,
    max_workers=min(2, multiprocessing.cpu_count()),
    spot_diameter=None,
):
    """
    Creates a Pseudovisium output directory containing binned spatial transcriptomics data.

    Args:
        output_path (str): Base path for output directory.
        hexagon_counts (scipy.sparse.csr_matrix): Matrix of binned counts.
        unique_hexagons (np.ndarray): Array of unique bin coordinates.
        unique_features (np.ndarray): Array of unique features/genes.
        hexagon_cell_counts (pd.DataFrame): DataFrame of binned cell counts.
        hexagon_quality (dict): Dictionary of bin quality metrics.
        probe_quality (dict): Dictionary of probe quality metrics.
        cell_id_colname (str): Name of the cell ID column.
        img_file_path (str, optional): Path to tissue image file.
        shift_to_positive (bool, optional): Whether to shift coordinates to positive values.
        project_name (str, optional): Name of the project subfolder.
        alignment_matrix_file (str, optional): Path to image alignment matrix.
        image_pixels_per_um (float, optional): Image resolution in pixels per micrometer.
        bin_size (int, optional): Size of the spatial bins.
        tissue_hires_scalef (float, optional): Scaling factor for high-res tissue image.
        pixel_to_micron (bool, optional): Whether to convert pixel coordinates to microns.
        max_workers (int, optional): Maximum number of parallel processes.
        spot_diameter (float, optional): Diameter for Visium-like spot array simulation.

    Notes:
        Creates a directory structure compatible with spatial transcriptomics analysis tools,
        including:
        - Binned count matrix in Matrix Market and h5 formats
        - Spatial coordinates and metadata
        - Tissue images (if provided)
        - Quality metrics (if calculated)
    """

    # to path, create a folder called pseudovisium
    folderpath = output_path + "/pseudovisium/" + project_name

    spot = True if spot_diameter != None else False
    if spot:
        print(
            "Visium-like array structure is being built rather than hexagonal tesselation!!!\n"
        )

    ############################################## ##############################################
    # see https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    # https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs

    scalefactors = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "tissue_lowres_scalef": tissue_hires_scalef / 10,
        "fiducial_diameter_fullres": 0,
        "hexagon_diameter": 2 * bin_size,
    }

    if spot:
        scalefactors["spot_diameter_fullres"] = spot_diameter * image_pixels_per_um
    else:
        scalefactors["spot_diameter_fullres"] = 2 * bin_size * image_pixels_per_um

    print("Creating scalefactors_json.json file in spatial folder.\n")
    with open(folderpath + "/spatial/scalefactors_json.json", "w") as f:
        json.dump(scalefactors, f)
    ############################################## ##############################################

    x, y, x_, y_, contain = [], [], [], [], []
    for i, hexagon in enumerate(unique_hexagons):
        # convert hexagon back to tuple from its string form
        hexagon = eval(hexagon)
        x_.append((hexagon[0] + bin_size) // (2 * bin_size))
        y_.append(hexagon[1] // (1.73205 * bin_size))
        x.append(hexagon[0])
        y.append(hexagon[1])
        contain.append(1 if hexagon_counts[i].sum() > bin_size else 0)

    ############################################## ##############################################
    barcodes = ["hexagon_{}".format(i) for i in range(1, len(unique_hexagons) + 1)]
    barcodes_table = pd.DataFrame({"barcode": barcodes})
    # save to pseudo visium root
    barcodes_table.to_csv(
        folderpath + "/barcodes.tsv", sep="\t", index=False, header=False
    )

    print("Creating barcodes.tsv.gz file.\n")
    with open(folderpath + "/barcodes.tsv", "rb") as f_in:
        with gzip.open(folderpath + "/barcodes.tsv.gz", "wb") as f_out:
            f_out.writelines(f_in)
    ############################################## ##############################################
    hexagon_table = pd.DataFrame(
        zip(
            barcodes,
            contain,
            y_,  # ideally in microns
            x_,  # ideally in microns
            [int(image_pixels_per_um * a) for a in y],  # in pixel units
            [int(image_pixels_per_um * a) for a in x],  # in pixel units
        ),
        columns=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    )

    if shift_to_positive:
        min_array_row = hexagon_table["array_row"].min()
        min_array_col = hexagon_table["array_col"].min()
        min_pxl_row = hexagon_table["pxl_row_in_fullres"].min()
        min_pxl_col = hexagon_table["pxl_col_in_fullres"].min()

        if min_array_row < 0:
            hexagon_table["array_row"] -= min_array_row
        if min_array_col < 0:
            hexagon_table["array_col"] -= min_array_col
        if min_pxl_row < 0:
            hexagon_table["pxl_row_in_fullres"] -= min_pxl_row
        if min_pxl_col < 0:
            hexagon_table["pxl_col_in_fullres"] -= min_pxl_col

    print("Creating tissue_positions_list.csv file in spatial folder.\n")
    hexagon_table.to_csv(
        folderpath + "/spatial/tissue_positions_list.csv", index=False, header=False
    )

    ############################################## ##############################################

    # if hexagon_cell_counts is pandas df
    if hexagon_cell_counts.empty:
        print("No cell information provided. Skipping cell information files.\n")
    else:
        print("Creating pv_cell_hex.csv file in spatial folder.\n")
        hexagon_cell_counts = hexagon_cell_counts[
            [cell_id_colname, "hexagon_id", "counts"]
        ]
        # add 1 to hexagon_ids
        hexagon_cell_counts["hexagon_id"] = hexagon_cell_counts["hexagon_id"] + 1
        # save csv
        hexagon_cell_counts.to_csv(
            folderpath + "/spatial/pv_cell_hex.csv", index=False, header=False
        )

    if hexagon_quality == {}:
        print("No quality information provided. Skipping quality information files.\n")
    else:
        print("Creating quality_per_hexagon.csv file in spatial folder.\n")
        with open(
            folderpath + "/spatial/quality_per_hexagon.csv", "w", newline=""
        ) as f:
            writer = csv.writer(f)
            for hexagon, quality_dict in hexagon_quality.items():
                try:
                    hexagon_index = np.where(unique_hexagons == str(hexagon))[0][0]
                    writer.writerow(
                        [hexagon_index + 1, quality_dict["mean"], quality_dict["count"]]
                    )
                except IndexError as e:
                    print(
                        f"Error: Unable to find hexagon '{hexagon}' in unique_hexagons.\n"
                    )
                    print(f"Error details: {str(e)}\n")
                    print(
                        f"Skipping quality measurement for hexagon '{hexagon}' with mean {quality_dict['mean']} and count {quality_dict['count']}.\n"
                    )

    ############################################## ##############################################
    if probe_quality == {}:
        print("No quality information provided. Skipping quality information files.\n")
    else:
        print("Creating quality_per_probe.csv file in spatial folder.\n")

        with open(folderpath + "/spatial/quality_per_probe.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for probe, quality_dict in probe_quality.items():
                writer.writerow([probe, quality_dict["mean"], quality_dict["count"]])

    ############################################## ##############################################

    features = unique_features
    # Create a list of rows with repeated features and 'Gene Expression' column
    rows = [[feature, feature, "Gene Expression"] for feature in features]

    print("Creating features.tsv.gz file.\n")
    with open(
        folderpath + "/features.tsv", "wt", newline="", encoding="utf-8"
    ) as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerows(rows)

    # Create a features.tsv.gz file
    with open(folderpath + "/features.tsv", "rb") as f_in, gzip.open(
        folderpath + "/features.tsv.gz", "wb"
    ) as f_out:
        f_out.writelines(f_in)

    ############################################## ##############################################

    print("Putting together the matrix.mtx file\n")

    print(f"Total matrix count: {hexagon_counts.sum()}\n")
    print(f"Number of unique hexagons: {len(barcodes)}\n")

    print("Creating matrix.mtx.gz file.\n")
    with open(folderpath + "/matrix.mtx", "wb") as f:
        scipy.io.mmwrite(
            f,
            hexagon_counts.T,
            comment='metadata_json: {"software_version": "Pseudovisium", "format_version": 1}\n',
        )

    with open(folderpath + "/matrix.mtx", "rb") as f_in, gzip.open(
        folderpath + "/matrix.mtx.gz", "wb"
    ) as f_out:
        f_out.writelines(f_in)

    ############################################## ##############################################

    print("Putting together the filtered_feature_bc_matrix.h5 file\n")

    # Create AnnData object from sparse matrix and barcodes/features
    adata = sc.AnnData(
        X=hexagon_counts,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=features),
    )

    # Write AnnData object to 10X-formatted h5 file
    write_10X_h5(adata, folderpath + "/filtered_feature_bc_matrix.h5")

    ############################################## ##############################################

    # check if alignment matrix file is given
    if alignment_matrix_file:
        print(
            "Alignment matrix found and will be used to create tissue_hires_image.png and tissue_lowres_image.png files in spatial folder.\n"
        )
        M = pd.read_csv(alignment_matrix_file, header=None, index_col=None).to_numpy()
    else:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Load the H&E image
    if img_file_path:
        print("Image provided at {0}\n".format(img_file_path))
        # if img_filepath is tiff
        if img_file_path.endswith(".tiff") or img_file_path.endswith(".tif"):
            image = tifffile.imread(img_file_path)
        # elif png
        elif img_file_path.endswith(".png"):
            image = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)
        # resizing the image according to tissue_hires_scalef, but such that it also satisfies the incoming
        # scaling factor from the alignment matrix

        if pixel_to_micron:
            M[0, 0] = (
                1 / M[0, 0]
            )  # Update x-scale to 1 because rescaling already done here
            M[1, 1] = (
                1 / M[1, 1]
            )  # Update y-scale to 1 because rescaling already done here
            width = int(image.shape[1] * tissue_hires_scalef)  # * M[0, 0])  # New width
            height = int(
                image.shape[0] * tissue_hires_scalef
            )  # * M[0, 0])  # New height
            dim = (width, height)
            # resize image
            resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # update translation
            M[0, 2] *= tissue_hires_scalef * M[0, 1]
            M[1, 2] *= tissue_hires_scalef * M[0, 1]

            M[0, 2] = -M[0, 2]
            M[1, 2] = -M[1, 2]

        else:  # normal scenario

            width = int(image.shape[1] * tissue_hires_scalef)  # New width
            height = int(image.shape[0] * tissue_hires_scalef)  # New height
            dim = (width, height)
            # resize image
            resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # update translation
            M[
                0, 2
            ] *= tissue_hires_scalef  # * max(M[0, 0], M[1, 1])  # Update x-translation
            M[
                1, 2
            ] *= tissue_hires_scalef  # * max(M[0, 0], M[1, 1])  # Update y-translation

        max_dim = max(resized_img.shape)
        max_stretch = max(M[0, 0], M[1, 1], M[1, 0], M[0, 1]) * 1.2
        # Apply the transformation
        new_width = int(max_dim * max_stretch)
        new_height = int(max_dim * max_stretch)
        image = cv2.warpAffine(resized_img, M[:2], (new_width, new_height))

        # and to 2%
        scale2 = 0.1
        width = int(image.shape[1] * scale2)
        height = int(image.shape[0] * scale2)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # finally if image is a single channel, convert to three channels, because some packages will expect that
        # when importing.
        dims = len(image.shape)
        # if 2, then triple the image to make it three channels
        if dims == 2:
            image = np.array([image, image, image])
            # change order of axis
            image = np.moveaxis(image, 0, -1)

            resized = np.array([resized, resized, resized])
            # change order of axis
            resized = np.moveaxis(resized, 0, -1)

        print("Creating tissue_hires_image.png file in spatial folder.\n")
        # save as 8bit
        cv2.imwrite(
            folderpath + "/spatial/tissue_hires_image.png", image / np.max(image) * 255
        )
        cv2.imwrite(
            folderpath + "/spatial/tissue_lowres_image.png",
            resized / np.max(resized) * 255,
        )

    else:
        # output a blank image
        # Create a white background image with tissue positions
        print(
            "No image file provided. Drawing tissue positions on a white background.\n"
        )

        # Extract pixel coordinates from hexagon_table
        pixel_coords = np.array(
            [
                [int(row * tissue_hires_scalef), int(col * tissue_hires_scalef)]
                for row, col in zip(
                    hexagon_table["pxl_row_in_fullres"],
                    hexagon_table["pxl_col_in_fullres"],
                )
            ]
        )

        # Find the maximum dimensions of the pixel coordinates
        max_x = np.max(pixel_coords[:, 1])
        max_y = np.max(pixel_coords[:, 0])

        # Create a white background image
        image = np.ones((max_y + 100, max_x + 100, 3), dtype=np.uint8) * 255

        # Draw purple dots at the tissue positions
        for coord in pixel_coords:
            cv2.circle(image, tuple(coord[::-1]), 5, (255, 0, 255), -1)

        # Save the high-resolution image
        cv2.imwrite(folderpath + "/spatial/tissue_hires_image.png", image)

        # Resize the image for the low-resolution version
        scale2 = 0.1
        width = int(image.shape[1] * scale2)
        height = int(image.shape[0] * scale2)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Save the low-resolution image
        cv2.imwrite(folderpath + "/spatial/tissue_lowres_image.png", resized)


#####################
###############  Visium HD/Slide-seq helper functions
###################################################################################################################################################
def read_files(folder, technology):
    """
    Reads spatial transcriptomics data files based on the technology platform.

    Args:
        folder (str): Path to the data folder.
        technology (str): Technology platform identifier:
            - "Visium_HD"/"VisiumHD"/"Visium HD": Reads Visium HD format files
            - "Curio": Reads Curio format files (.h5ad)
            - "Zarr": Reads SpatialData Zarr stores

    Returns:
        Union[Tuple[dict, pd.DataFrame, anndata.AnnData], anndata.AnnData, spatialdata.SpatialData]:
            - For Visium HD: (scalefactors, tissue positions, feature-barcode matrix)
            - For Curio: AnnData object
            - For Zarr: SpatialData object
    """

    if (
        (technology == "Visium_HD")
        or (technology == "VisiumHD")
        or (technology == "Visium HD")
    ):
        scalefactors = json.load(open(folder + "/spatial/scalefactors_json.json"))
        tissue_pos = pd.read_parquet(folder + "/spatial/tissue_positions.parquet")
        fb_matrix = sc.read_10x_h5(folder + "/filtered_feature_bc_matrix.h5")
        return scalefactors, tissue_pos, fb_matrix
    elif technology == "Curio":
        # find file ending with .h5ad in folder
        folder_files = os.listdir(folder)
        h5ad_file = [file for file in folder_files if file.endswith(".h5ad")][0]
        adata = sc.read_h5ad(folder + h5ad_file)
        return adata
    elif technology == "Zarr":

        sdata = sd.read_zarr(folder)

        return sdata


def asdata_to_df(
    asdata,
    technology,
    tissue_pos=None,
    scalefactors=None,
    x_col=None,
    y_col=None,
    sd_table_id=None,
):
    """
    Converts various spatial data objects to a standardized DataFrame format.

    Args:
        asdata (Union[anndata.AnnData, spatialdata.SpatialData]): Input data object.
        technology (str): Technology platform identifier:
            - "Visium_HD"/"VisiumHD"/"Visium HD": Processes Visium HD format
            - "Curio": Processes Curio format
            - "SpatialData": Processes SpatialData format
            - "AnnData": Processes AnnData format
        tissue_pos (pd.DataFrame, optional): Tissue positions for Visium HD data.
        scalefactors (dict, optional): Scale factors for Visium HD data.
        x_col (str, optional): Name of x-coordinate column in Curio data.
        y_col (str, optional): Name of y-coordinate column in Curio data.
        sd_table_id (str, optional): Table identifier in SpatialData object.

    Returns:
        Union[Tuple[pd.DataFrame, float], pd.DataFrame]:
            - For Visium HD: (DataFrame, image resolution in pixels per micron)
            - For Curio: (DataFrame, minimal distance between points)
            - For SpatialData/AnnData: DataFrame with standardized columns:
                - barcode: Spot/cell identifier
                - gene: Gene/feature name
                - count: Expression count
                - x: X-coordinate
                - y: Y-coordinate
    """

    if (
        (technology == "Visium_HD")
        or (technology == "VisiumHD")
        or (technology == "Visium HD")
    ):
        # get image resolution of the hires image
        image_resolution = (
            scalefactors["microns_per_pixel"] / scalefactors["tissue_hires_scalef"]
        )
        # change to pixel per micron
        image_resolution = 1 / image_resolution
        # tissue_pos keep barcode, pxl_row_in_fullres, pxl_col_in_fullres
        tissue_pos = tissue_pos[["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"]]
        # Convert the AnnData matrix to a sparse matrix (CSR format)
        X = scipy.sparse.csr_matrix(asdata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()

        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame(
            {
                "barcode": asdata.obs_names[row_indices],
                "gene": asdata.var_names[col_indices],
                "count": X.data,
            }
        )

        # join with tissue_pos
        df = pd.merge(df, tissue_pos, on="barcode")

        # extract microns_per_pixel
        microns_per_pixel = scalefactors["microns_per_pixel"]
        # convert pxl_row_in_fullres and pxl_col_in_fullres to microns
        df["y"] = df["pxl_row_in_fullres"] * microns_per_pixel
        df["x"] = df["pxl_col_in_fullres"] * microns_per_pixel
        # remove pxl_row_in_fullres and pxl_col_in_fullres
        df.drop(["pxl_row_in_fullres", "pxl_col_in_fullres"], axis=1, inplace=True)

        # Write the DataFrame to a CSV file
        return df, image_resolution
    elif technology == "Curio":
        # get scales

        # keep only the columns x,y
        obs = asdata.obs[[x_col, y_col]]
        # rename these cols to x and y
        obs.columns = ["x", "y"]
        scale = np.round(
            min(
                [abs(x1 - x2) for x1 in obs["y"] for x2 in [obs["y"][500]] if x1 != x2]
            ),
            3,
        )  # should be 10um

        obs["barcode"] = obs.index
        X = scipy.sparse.csr_matrix(asdata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()
        genes = (
            asdata.var.index
            if (asdata.var.index[0] != 0) and (asdata.var.index[0] != "0")
            else asdata.var[asdata.var.columns[0]]
        )
        genes = genes[col_indices]
        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame(
            {"barcode": asdata.obs_names[row_indices], "gene": genes, "count": X.data}
        )
        # join with tissue_pos
        df = pd.merge(df, obs, on="barcode")
        return df, scale

    elif technology == "SpatialData":

        adata = asdata.tables[sd_table_id].copy()
        adata.obs["x"] = adata.obsm["spatial"][:, 0]
        adata.obs["y"] = adata.obsm["spatial"][:, 1]

        # keep only the columns x,y
        obs = adata.obs[["x", "y"]]

        obs["barcode"] = obs.index
        X = scipy.sparse.csr_matrix(adata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()
        genes = (
            adata.var.index
            if (adata.var.index[0] != 0) and (adata.var.index[0] != "0")
            else adata.var[adata.var.columns[0]]
        )
        genes = genes[col_indices]
        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame(
            {"barcode": adata.obs_names[row_indices], "gene": genes, "count": X.data}
        )
        # join with tissue_pos
        df = pd.merge(df, obs, on="barcode")

        return df

    elif technology == "AnnData":

        adata = asdata.copy()
        adata.obs["x"] = adata.obsm["spatial"][:, 0]
        adata.obs["y"] = adata.obsm["spatial"][:, 1]

        # keep only the columns x,y
        obs = adata.obs[["x", "y"]]

        obs["barcode"] = obs.index
        X = scipy.sparse.csr_matrix(adata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()
        genes = (
            adata.var.index
            if (adata.var.index[0] != 0) and (adata.var.index[0] != "0")
            else adata.var[adata.var.columns[0]]
        )
        genes = genes[col_indices]
        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame(
            {"barcode": adata.obs_names[row_indices], "gene": genes, "count": X.data}
        )
        # join with tissue_pos
        df = pd.merge(df, obs, on="barcode")

        return df


def anndata_to_transcripts_pq(
    folder_or_object, output, technology, x_col=None, y_col=None, sd_table_id=None
):
    """
    Converts various spatial data formats to a standardized Parquet file.

    Args:
        folder_or_object (Union[str, anndata.AnnData, spatialdata.SpatialData]):
            Input data source - either a path to data folder or a data object.
        output (str): Path where the output Parquet file will be saved.
        technology (str): Technology platform identifier:
            - "Visium_HD"/"VisiumHD"/"Visium HD": Processes Visium HD format
            - "Curio": Processes Curio format
            - "Zarr": Processes Zarr stores
            - "SpatialData": Processes SpatialData objects
            - "AnnData": Processes AnnData objects
        x_col (str, optional): Name of x-coordinate column in Curio data.
        y_col (str, optional): Name of y-coordinate column in Curio data.
        sd_table_id (str, optional): Table identifier in SpatialData object.

    Returns:
        Optional[float]:
            - For Visium HD: Image resolution in pixels per micron
            - For Curio: Minimal distance between points
            - For others: None

    Notes:
        Creates a standardized Parquet file containing:
        - barcode: Spot/cell identifier
        - gene: Gene/feature name
        - count: Expression count
        - x: X-coordinate
        - y: Y-coordinate
    """

    if (
        (technology == "Visium_HD")
        or (technology == "VisiumHD")
        or (technology == "Visium HD")
    ):
        scalefactors, tissue_pos, fb_matrix = read_files(folder_or_object, technology)
        df, image_resolution = asdata_to_df(
            asdata=fb_matrix,
            technology=technology,
            tissue_pos=tissue_pos,
            scalefactors=scalefactors,
        )
        # create parquet file
        df.to_parquet(output, index=False)
        return image_resolution
    elif technology == "Curio":
        adata = read_files(folder_or_object, technology)
        df, scale = asdata_to_df(
            asdata=adata, technology=technology, x_col=x_col, y_col=y_col
        )
        df.to_parquet(output, index=False)
        return scale

    elif technology == "Zarr":
        asdata = read_files(folder_or_object, technology)
        df = asdata_to_df(
            asdata=asdata, technology="SpatialData", sd_table_id=sd_table_id
        )
        df.to_parquet(output, index=False)
        return None

    elif technology == "SpatialData":
        df = asdata_to_df(
            asdata=folder_or_object, technology=technology, sd_table_id=sd_table_id
        )
        df.to_parquet(output, index=False)
        return None

    elif technology == "AnnData":
        df = asdata_to_df(
            asdata=folder_or_object, technology=technology, sd_table_id=sd_table_id
        )
        df.to_parquet(output, index=False)
        return None


######### Main function to generate pseudovisium output ############################################################################################################


def generate_pv(
    csv_file=None,
    img_file_path=None,
    shift_to_positive=False,
    bin_size=100,
    output_path=None,
    batch_size=1000000,
    alignment_matrix_file=None,
    project_name="project",
    image_pixels_per_um=1,
    tissue_hires_scalef=0.2,
    technology="Xenium",
    feature_colname="feature_name",
    x_colname="x_location",
    y_colname="y_location",
    cell_id_colname="None",
    quality_colname="qv",
    pixel_to_micron=False,
    max_workers=min(2, multiprocessing.cpu_count()),
    quality_filter=False,
    count_colname="NA",
    folder_or_object=None,
    smoothing=False,
    quality_per_hexagon=False,
    quality_per_probe=False,
    h5_x_colname="x",
    h5_y_colname="y",
    coord_to_um_conversion=1,
    spot_diameter=None,
    hex_square="hex",
    sd_table_id=None,
):
    """
    Main function to generate Pseudovisium output from various spatial transcriptomics data formats.

    Args:
        csv_file (str, optional): Path to input data file (CSV, Parquet, or gzipped CSV).
        img_file_path (str, optional): Path to tissue image file.
        shift_to_positive (bool, optional): Whether to shift coordinates to positive values.
        bin_size (float, optional): Size of the spatial bins.
        output_path (str, optional): Base path for output directory.
        batch_size (int, optional): Number of rows to process per batch.
        alignment_matrix_file (str, optional): Path to image alignment matrix.
        project_name (str, optional): Name of the project subfolder.
        image_pixels_per_um (float, optional): Image resolution in pixels per micrometer.
        tissue_hires_scalef (float, optional): Scaling factor for high-res tissue image.
        technology (str, optional): Technology platform name.
        feature_colname (str, optional): Name of the feature/gene column.
        x_colname (str, optional): Name of the x-coordinate column.
        y_colname (str, optional): Name of the y-coordinate column.
        cell_id_colname (str, optional): Name of the cell ID column.
        quality_colname (str, optional): Name of the quality score column.
        pixel_to_micron (bool, optional): Whether to convert pixel coordinates to microns.
        max_workers (int, optional): Maximum number of parallel processes.
        quality_filter (bool, optional): Whether to filter rows based on quality score.
        count_colname (str, optional): Name of the count column.
        folder_or_object (str/object, optional): Path to data folder or data object.
        smoothing (bool/float, optional): Whether/how much to smooth the counts.
        quality_per_hexagon (bool, optional): Whether to calculate quality metrics per bin.
        quality_per_probe (bool, optional): Whether to calculate quality metrics per probe.
        h5_x_colname (str, optional): Name of x-coordinate column in h5 files.
        h5_y_colname (str, optional): Name of y-coordinate column in h5 files.
        coord_to_um_conversion (float, optional): Conversion factor from coordinates to micrometers.
        spot_diameter (float, optional): Diameter for Visium-like spot array simulation.
        hex_square (str, optional): Shape of spatial bins: "hex" or "square".
        sd_table_id (str, optional): Table identifier in SpatialData object.

    Notes:
        Supports multiple input formats:
        - Raw data files (CSV, Parquet, gzipped CSV)
        - Visium HD data
        - Curio data
        - SpatialData/AnnData objects
        - Various commercial platforms (Xenium, Vizgen, CosMx, etc.)

        Creates a complete output directory with:
        - Binned count matrices
        - Spatial coordinates
        - Quality metrics
        - Tissue images (if provided)
        - Configuration and metadata files
    """
    try:
        output = (
            subprocess.check_output(["pip", "freeze"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        version = [x for x in output if "Pseudovisium" in x]
        date = str(datetime.datetime.now().date())
        print("#####################\n")
        print("You are using version: ", version)
        print("Date: ", date)
        print("#####################\n")
    except:
        output = (
            subprocess.check_output(["pip3", "freeze"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        version = [x for x in output if "Pseudovisium" in x]
        date = str(datetime.datetime.now().date())
        print("#####################\n")
        print("You are using version: ", version)
        print("Date: ", date)
        print("#####################\n")
    print("#####################\n")
    print(
        """This is Pseudovisium, a software that compresses imaging-based spatial transcriptomics files using hexagonal binning of the data.
            Please cite: Kövér B, Vigilante A. https://www.biorxiv.org/content/10.1101/2024.07.23.604776v1\n
            """
    )
    print("#####################\n")

    #try: commented out for troubleshooting

    start = time.time()

    # to path, create a folder called pseudovisium
    folderpath = output_path + "/pseudovisium/" + project_name

    try:
        print(
            "Creating pseudovisium folder in output path:{0}\n".format(folderpath)
        )

        if os.path.exists(output_path + "/pseudovisium/"):
            print(
                "Using already existing folder: {0}\n".format(
                    output_path + "/pseudovisium/"
                )
            )
        else:
            os.mkdir(output_path + "/pseudovisium/")

        if os.path.exists(folderpath):
            shutil.rmtree(folderpath)

        os.mkdir(folderpath)

        if os.path.exists(folderpath + "/spatial/"):
            print(
                "Using already existing folder: {0}\n".format(
                    folderpath + "/spatial/"
                )
            )
        else:
            os.mkdir(folderpath + "/spatial/")
    except:
        pass


    # check if folder_or_object is of type AnnData. If so, change technology to AnnData
    if isinstance(folder_or_object, ad.AnnData):
        technology = "AnnData"
    # if SpatialData object, change technology to SpatialData
    if isinstance(folder_or_object, sd.SpatialData):
        technology = "SpatialData"
    # if folderpath ends with .zarr or .Zarr, change technology to Zarr
    if folderpath.endswith(".zarr") or folderpath.endswith(".Zarr"):
        technology = "Zarr"

    if (
        (technology == "Visium_HD")
        or (technology == "VisiumHD")
        or (technology == "Visium HD")
    ):
        print(
            "Technology is Visium_HD. Generating transcripts.parquet file from Visium HD files.\n"
        )
        # Automatically calculating image_pixels_per_um from the scalefactors_json.json file
        image_pixels_per_um = anndata_to_transcripts_pq(
            folder_or_object, folder_or_object + "/transcripts.parquet", technology
        )
        csv_file = folderpath + "/transcripts.parquet"

    if technology == "Curio":
        print(
            "Technology is Curio. Generating transcripts.parquet file from Curio files.\n"
        )
        smoothing_scale = anndata_to_transcripts_pq(
            folder_or_object,
            folderpath + "/transcripts.parquet",
            technology,
            x_col=h5_x_colname,
            y_col=h5_y_colname,
        )
        csv_file = folderpath + "/transcripts.parquet"
        print("Smoothing defaults to : {0}".format(smoothing_scale / 4))
        smoothing = smoothing_scale / 4

    if technology == "Zarr":
        print(
            "Technology is unclear, but you passed a Zarr file. Generating transcripts.parquet file from Zarr files.\n"
        )
        anndata_to_transcripts_pq(
            folder_or_object,
            folderpath + "/transcripts.parquet",
            technology,
            sd_table_id=sd_table_id,
        )
        csv_file = folderpath + "/transcripts.parquet"

    if (technology == "SpatialData") or (technology == "AnnData"):

        print(
            "Technology is unclear, but you passed a SpatialData/AnnData object. Generating transcripts.parquet file.\n"
        )
        anndata_to_transcripts_pq(
            folder_or_object,
            folderpath + "/transcripts.parquet",
            technology,
            sd_table_id=sd_table_id,
        )
        csv_file = folderpath + "/transcripts.parquet"

    if technology == "Xenium":
        print("Technology is Xenium. Going forward with default column names.\n")
        x_colname = "x_location"
        y_colname = "y_location"
        feature_colname = "feature_name"
        cell_id_colname = "cell_id"
        quality_colname = "qv"
        count_colname = "NA"
        # coord_to_um_conversion = 1

    elif technology == "Vizgen":
        print("Technology is Vizgen. Going forward with default column names.\n")
        x_colname = "global_x"
        y_colname = "global_y"
        feature_colname = "gene"
        # cell_id_colname = "barcode_id"
        count_colname = "NA"
        # coord_to_um_conversion = 1

    elif (technology == "Nanostring") or (technology == "CosMx"):
        print(
            "Technology is Nanostring. Going forward with default column names.\n"
        )
        x_colname = "x_global_px"
        y_colname = "y_global_px"
        feature_colname = "target"
        cell_id_colname = "cell"
        count_colname = "NA"
        if coord_to_um_conversion == 1:
            print(
                "Knowing CosMx, we are setting coord_to_um_conversion to 0.12028, which is likely better than the default 1.\n"
            )
            coord_to_um_conversion = 0.12028
        # see ref https://smi-public.objects.liquidweb.services/cosmx-wtx/Pancreas-CosMx-ReadMe.html
        # https://nanostring.com/wp-content/uploads/2023/09/SMI-ReadMe-BETA_humanBrainRelease.html
        # Whereas old smi output seems to be 0.18
        # https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html

    elif (
        (technology == "Visium_HD")
        or (technology == "VisiumHD")
        or (technology == "Visium HD")
    ):
        print(
            "Technology is Visium_HD. Going forward with pseudovisium processed colnames.\n"
        )
        x_colname = "x"
        y_colname = "y"
        feature_colname = "gene"
        cell_id_colname = "barcode"
        count_colname = "count"
        # coord_to_um_conversion = 1

    elif technology == "seqFISH":
        print("Technology is seqFISH. Going forward with default column names.\n")
        x_colname = "x"
        y_colname = "y"
        feature_colname = "name"
        cell_id_colname = "cell"
        count_colname = "NA"
        # coord_to_um_conversion = 1

    elif technology == "Curio":
        print("Technology is Curio. Going forward with default column names.\n")
        x_colname = "x"
        y_colname = "y"
        feature_colname = "gene"
        cell_id_colname = "barcode"
        count_colname = "count"
        # coord_to_um_conversion = 1

    elif (
        (technology == "Zarr")
        or technology == ("SpatialData")
        or (technology == "AnnData")
    ):
        print(
            "Technology is Zarr/SpatialData/AnnData. Going forward with default column names.\n"
        )
        x_colname = "x"
        y_colname = "y"
        feature_colname = "gene"
        cell_id_colname = "barcode"
        count_colname = "count"

    else:
        print("Technology not recognized. Going forward with set column names.\n")

    (
        hexagon_counts,
        unique_hexagons,
        unique_features,
        hexagon_cell_counts,
        hexagon_quality,
        probe_quality,
    ) = process_csv_file(
        csv_file,
        bin_size,
        batch_size,
        feature_colname,
        x_colname,
        y_colname,
        cell_id_colname,
        quality_colname=quality_colname,
        max_workers=max_workers,
        quality_filter=quality_filter,
        count_colname=count_colname,
        smoothing=smoothing,
        quality_per_hexagon=quality_per_hexagon,
        quality_per_probe=quality_per_probe,
        coord_to_um_conversion=coord_to_um_conversion,
        spot_diameter=spot_diameter,
        hex_square=hex_square,
    )

    # Create Pseudovisium output
    create_pseudovisium(
        output_path=output_path,
        hexagon_counts=hexagon_counts,
        unique_hexagons=unique_hexagons,
        unique_features=unique_features,
        hexagon_cell_counts=hexagon_cell_counts,
        hexagon_quality=hexagon_quality,
        probe_quality=probe_quality,
        cell_id_colname=cell_id_colname,
        img_file_path=img_file_path,
        shift_to_positive=shift_to_positive,
        project_name=project_name,
        alignment_matrix_file=alignment_matrix_file,
        image_pixels_per_um=image_pixels_per_um,
        bin_size=bin_size,
        tissue_hires_scalef=tissue_hires_scalef,
        pixel_to_micron=pixel_to_micron,
        max_workers=max_workers,
        spot_diameter=spot_diameter,
    )

    # save all arguments in a json file called arguments.json
    print("Creating arguments.json file in output path.\n")
    if type(folder_or_object) != str:
        folder_or_object_entry = str(type(folder_or_object))
    else:
        folder_or_object_entry = folder_or_object

    arguments = {
        "csv_file": csv_file,
        "img_file_path": img_file_path,
        "bin_size": bin_size,
        "output_path": output_path,
        "batch_size": batch_size,
        "alignment_matrix_file": alignment_matrix_file,
        "project_name": project_name,
        "image_pixels_per_um": image_pixels_per_um,
        "tissue_hires_scalef": tissue_hires_scalef,
        "technology": technology,
        "feature_colname": feature_colname,
        "x_colname": x_colname,
        "y_colname": y_colname,
        "cell_id_colname": cell_id_colname,
        "pixel_to_micron": pixel_to_micron,
        "quality_colname": quality_colname,
        "quality_filter": quality_filter,
        "count_colname": count_colname,
        "smoothing": smoothing,
        "quality_per_hexagon": quality_per_hexagon,
        "quality_per_probe": quality_per_probe,
        "max_workers": max_workers,
        "folder_or_object": folder_or_object_entry,  # this is because of course in case it is an object we cannot write it to a json
        "h5_x_colname": h5_x_colname,
        "h5_y_colname": h5_y_colname,
        "coord_to_um_conversion": coord_to_um_conversion,
        "spot_diameter": spot_diameter,
        "hex_square": hex_square,
    }

    with open(folderpath + "/arguments.json", "w") as f:
        json.dump(arguments, f)

    end = time.time()
    print(f"Time taken: {end - start} seconds\n")
    #except Exception as e:
    #    print("Error: Unable to generate Pseudovisium output.")
    #    print(f"Error details: {str(e)}")


def adata_to_adata(adata, bin_size, bin_type="hex"):
    """
    Performs spatial binning (hexagonal or square) on an AnnData object based on spatial coordinates.

    Args:
        adata (AnnData): Input AnnData object containing spatial information in adata.obsm["spatial"].
        bin_size (float): Size of the spatial bin:
            - For hexagonal bins: distance from center to middle of edge (apothem)
            - For square bins: length of a side
        bin_type (str): Type of binning: 'hex' for hexagonal or 'square' for square bins.

    Returns:
        AnnData: New AnnData object with binned data, containing:
            - Aggregated expression matrix
            - Bin coordinates in obsm["spatial"]
            - Number of cells per bin in obs["n_cells"]
            - Original metadata and variables
            - Additional binning information in uns
    """

    # Extract spatial coordinates
    coords = adata.obsm["spatial"]
    x, y = coords[:, 0], coords[:, 1]

    # Choose the appropriate binning function
    if bin_type == "hex":
        bin_func = closest_hex
    elif bin_type == "square":
        bin_func = closest_square
    else:
        raise ValueError("bin_type must be either 'hex' or 'square'")

    # Assign cells to bins
    bins = np.array(
        [bin_func(x_coord, y_coord, bin_size) for x_coord, y_coord in zip(x, y)]
    )

    # Create a DataFrame with bin assignments
    df = pd.DataFrame(
        {"bin_x": bins[:, 0], "bin_y": bins[:, 1], "original_index": range(adata.n_obs)}
    )

    # Group by bin and aggregate
    grouped = df.groupby(["bin_x", "bin_y"])["original_index"].apply(list)

    # Create new observation data
    new_obs = pd.DataFrame(index=[f"{bin_type}_bin_{i}" for i in range(len(grouped))])
    new_obs["n_cells"] = grouped.apply(len).values

    # Aggregate expression data
    new_X = scipy.sparse.lil_matrix((len(grouped), adata.n_vars))
    for i, (_, indices) in enumerate(grouped.items()):
        new_X[i] = adata.X[indices].sum(axis=0)
    new_X = new_X.tocsr()

    # Create new AnnData object
    new_adata = sc.AnnData(X=new_X, obs=new_obs, var=adata.var.copy())

    # Add spatial coordinates for the bins
    new_adata.obsm["spatial"] = np.array(grouped.index.tolist())

    # Add metadata
    new_adata.uns = adata.uns.copy()
    new_adata.uns["bin_size"] = bin_size
    new_adata.uns["bin_type"] = bin_type
    new_adata.uns["original_adata_shape"] = adata.shape

    return new_adata


def spatialdata_to_spatialdata(
    sdata, table_id, new_table_id=None, new_shapes_id=None, bin_size=50, bin_type="hex"
):
    """
    Converts spatial data from a SpatialData object to a new SpatialData object with binned data,
    creating both a binned table and corresponding spatial tessellation.

    Args:
        sdata (SpatialData): Input SpatialData object containing spatial data.
        table_id (str): Identifier for the table in sdata to be binned.
        new_table_id (str, optional): Identifier for the new binned table. Defaults to
            "{table_id}_PV_bins_{bin_size}_{bin_type}".
        new_shapes_id (str, optional): Identifier for the new tessellation shapes. Defaults to
            "{table_id}_tessellation_{bin_size}_{bin_type}".
        bin_size (float, optional): Size parameter for binning:
            - For hexagonal bins: distance from center to middle of edge (apothem)
            - For square bins: length of a side
        bin_type (str, optional): Type of binning: "hex" for hexagonal or "square".

    Notes:
        - Modifies the input SpatialData object in-place by adding:
            - New binned table with identifier new_table_id
            - New tessellation shapes with identifier new_shapes_id
        - Original data remains unchanged
        - Binned table includes spatial coordinates in obsm["spatial"]
        - Generated tessellation is stored as a ShapesModel
        - Location IDs and region information are automatically generated and stored
    """

    import geopandas as gpd
    from shapely.geometry import Polygon

    if new_table_id is None:
        new_table_id = table_id + "_PV_bins_" + str(bin_size) + "_" + bin_type
    if new_shapes_id is None:
        new_shapes_id = table_id + "_tessellation_" + str(bin_size) + "_" + bin_type

    # if a float introduced a . to these names, turn it into a _
    new_table_id = new_table_id.replace(".", "_")
    new_shapes_id = new_shapes_id.replace(".", "_")

    # extract anndata_table
    extracted_table = sdata.tables[table_id].copy()

    print("Performing spatial binning...\n")
    binned_table = adata_to_adata(extracted_table, bin_size=bin_size, bin_type=bin_type)
    print(
        "Spatial binning complete.\n Adding new table and tessellation to sdata object...\n"
    )
    coordinates = binned_table.obsm["spatial"]

    def create_tessellation(cx, cy, bin_size, bin_type):
        if bin_type == "hex":
            radius = np.sqrt(3 / 2) * bin_size
            # Generate the six points of the hexagon
            angles = np.linspace(0, 2 * np.pi, 7)  # 0 to 360 degrees (2π radians)
            # add 30 degrees to the angle
            angles = angles + np.pi / 6
            x_hex = cx + radius * np.cos(angles)
            y_hex = cy + radius * np.sin(angles)
            return Polygon(zip(x_hex, y_hex))
        if bin_type == "square":
            return Polygon(
                [
                    (cx - bin_size / 2, cy - bin_size / 2),
                    (cx + bin_size / 2, cy - bin_size / 2),
                    (cx + bin_size / 2, cy + bin_size / 2),
                    (cx - bin_size / 2, cy + bin_size / 2),
                    (cx - bin_size / 2, cy - bin_size / 2),
                ]
            )

    # Create hexagon polygons at each coordinate and store in a list
    tessellation = [
        create_tessellation(x, y, bin_size, bin_type) for x, y in coordinates
    ]
    # Create the GeoDataFrame
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"geometry": tessellation}), geometry="geometry"
    )
    # Set location_id as the named index
    gdf.index.name = "location_id"

    # add columns: location_id, region_id
    binned_table.uns["spatialdata_attrs"]["region_key"] = "region"
    binned_table.uns["spatialdata_attrs"]["instance_key"] = "location_id"
    binned_table.uns["spatialdata_attrs"]["region"] = new_shapes_id
    binned_table.obs["location_id"] = np.arange(binned_table.shape[0])
    binned_table.obs["region"] = np.array([new_shapes_id] * binned_table.shape[0])

    from spatialdata.models import ShapesModel, TableModel

    sdata.shapes[new_shapes_id] = ShapesModel.parse(gdf)
    sdata.tables[new_table_id] = TableModel.parse(binned_table)
    print("New table and tessellation added to sdata object.\n Done.\n")


def main():
    """
    The main function that parses command-line arguments and calls the generate_pv function.

    This function sets up the argument parser, processes the command-line arguments,
    and calls the generate_pv function with the provided parameters. It also handles
    the display of help information and verbose output if requested.
    """
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument(
        "--csv_file", "-c", type=str, help="CSV file path", default=None
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Output path", default="."
    )
    parser.add_argument("--bin_size", "-bs", type=float, help="Bin size", default=50.0)
    parser.add_argument(
        "--img_file_path", "-i", type=str, help="Image file path", default=None
    )
    parser.add_argument(
        "--shift_to_positive",
        "-stp",
        action="store_true",
        help="Shift columns, rows, and full-resolution pixel values to positive if any are negative",
    )
    parser.add_argument(
        "--alignment_matrix_file",
        "-am",
        type=str,
        help="Alignment matrix file path",
        default=None,
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, help="Batch size", default=1000000
    )
    parser.add_argument(
        "--project_name", "-p", type=str, help="Project name", default="project"
    )
    parser.add_argument(
        "--image_pixels_per_um",
        "-ppu",
        type=float,
        help="Image pixels per um",
        default=1,
    )  # changed from 1/0.2125 that was set for Xenium
    parser.add_argument(
        "--tissue_hires_scalef",
        "-ths",
        type=float,
        help="Tissue hires scale factor",
        default=0.2,
    )
    parser.add_argument(
        "--technology", "-t", type=str, help="Technology", default="Xenium"
    )
    parser.add_argument(
        "--feature_colname",
        "-fc",
        type=str,
        help="Feature column name",
        default="feature_name",
    )
    parser.add_argument(
        "--x_colname", "-xc", type=str, help="X column name", default="x_location"
    )
    parser.add_argument(
        "--y_colname", "-yc", type=str, help="Y column name", default="y_location"
    )
    parser.add_argument(
        "--cell_id_colname", "-cc", type=str, help="Cell ID column name", default="None"
    )
    parser.add_argument(
        "--pixel_to_micron", "-ptm", action="store_true", help="Convert pixel to micron"
    )
    parser.add_argument(
        "--quality_colname", "-qcol", type=str, help="Quality column name", default="qv"
    )
    parser.add_argument(
        "--count_colname", "-ccol", type=str, help="Count column name", default="NA"
    )
    parser.add_argument(
        "--smoothing", "-s", type=float, help="Smoothing factor", default=0.0
    )
    parser.add_argument(
        "--folder_or_object", "-foo", type=str, help="Folder or Object", default=None
    )
    parser.add_argument(
        "--mw",
        "--max_workers",
        type=int,
        help="Max workers",
        default=min(2, multiprocessing.cpu_count()),
    )
    parser.add_argument(
        "--quality_filter",
        "-qf",
        action="store_true",
        help="Filter out rows with quality score less than 20",
    )
    parser.add_argument(
        "--quality_per_hexagon",
        "-qph",
        action="store_true",
        help="Calculate quality per hexagon",
    )
    parser.add_argument(
        "--quality_per_probe",
        "-qpp",
        action="store_true",
        help="Calculate quality per probe",
    )
    parser.add_argument(
        "--h5_x_colname",
        "-h5x",
        type=str,
        help="X column name in h5ad file",
        default="x",
    )
    parser.add_argument(
        "--h5_y_colname",
        "-h5y",
        type=str,
        help="Y column name in h5ad file",
        default="y",
    )
    parser.add_argument(
        "--coord_to_um_conversion",
        "-ctu",
        type=float,
        help="Conversion factor from coordinates to microns",
        default=1.0,
    )
    parser.add_argument(
        "--spot_diameter", "-sd", type=float, help="Spot diameter", default=None
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print out script purpose and parameters",
    )

    # make sure to add verbose as well
    parser.add_argument("-help", action="store_true", help="Print out help information")

    parser.add_argument(
        "--hex_square",
        "-hex",
        type=str,
        help="Shape of observational unit",
        default="hex",
    )

    parser.add_argument(
        "--sd_table_id",
        "-sid",
        type=str,
        help="Table ID in SpatialData object",
        default=None,
    )

    args = parser.parse_args()

    if args.help:
        print(
            """\nThis is Pseudovisium, a software that compresses imaging-based spatial transcriptomics files using hexagonal binning of the data.
            Please cite: Kövér B, Vigilante A. https://www.biorxiv.org/content/10.1101/2024.07.23.604776v1\n
            """
        )
        parser.print_help()
        sys.exit(0)

    generate_pv(
        csv_file=args.csv_file,
        img_file_path=args.img_file_path,
        shift_to_positive=args.shift_to_positive,
        bin_size=args.bin_size,
        output_path=args.output_path,
        batch_size=args.batch_size,
        alignment_matrix_file=args.alignment_matrix_file,
        project_name=args.project_name,
        image_pixels_per_um=args.image_pixels_per_um,
        tissue_hires_scalef=args.tissue_hires_scalef,
        technology=args.technology,
        feature_colname=args.feature_colname,
        x_colname=args.x_colname,
        y_colname=args.y_colname,
        cell_id_colname=args.cell_id_colname,
        pixel_to_micron=args.pixel_to_micron,
        max_workers=args.mw,
        quality_colname=args.quality_colname,
        quality_filter=args.quality_filter,
        count_colname=args.count_colname,
        smoothing=args.smoothing,
        quality_per_hexagon=args.quality_per_hexagon,
        quality_per_probe=args.quality_per_probe,
        h5_x_colname=args.h5_x_colname,
        h5_y_colname=args.h5_y_colname,
        coord_to_um_conversion=args.coord_to_um_conversion,
        folder_or_object=args.folder_or_object,
        spot_diameter=args.spot_diameter,
        hex_square=args.hex_square,
        sd_table_id=args.sd_table_id,
    )

    print("End.\n")


if __name__ == "__main__":
    main()
