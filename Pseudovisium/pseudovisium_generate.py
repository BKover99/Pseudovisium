import csv
import numpy as np
import math
import pandas as pd
import cv2
import json
import gzip
import concurrent.futures
import os
import shutil
import tempfile
from tqdm import tqdm
import itertools
import argparse
import tifffile
import multiprocessing
import time
import scanpy as sc
import scipy.io
import h5py
import scipy.sparse
from pathlib import Path
import subprocess
import datetime


def delete_temporary_files():
    """
    Deletes temporary batch files and directories created by the script.

    This function searches for directories starting with "tmp_hexa" in the system's
    temporary directory and prompts the user to confirm deletion of these directories
    and their contents. If the user confirms, the directories are deleted.
    """

    temp_dir = tempfile.gettempdir()
    print(f"Searching for temporary batch files in: {temp_dir}")

    remaining_files = []
    for root, dirs, files in os.walk(temp_dir):
        for dir_name in dirs:
            if dir_name.startswith("tmp_hexa"):
                dir_path = os.path.join(root, dir_name)
                remaining_files.append(dir_path)

    if remaining_files:
        print(f"Found {len(remaining_files)} remaining temporary directories.")

        confirmation = input(
            "Do you want to delete these temporary directories and their contents? (y/n): "
        )
        if confirmation.lower() == "y":
            for dir_path in remaining_files:
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")
                except OSError as e:
                    print(f"Error deleting {dir_path}: {e}")
            print("Temporary directories deleted.")
        else:
            print("Deletion canceled.")
    else:
        print("No remaining temporary batch files found.")


def closest_hex(x, y, hexagon_size, spot_diameter=None):
    """
    Calculates the closest hexagon centroid to the given (x, y) coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        hexagon_size (float): The size of the hexagon.
        spot_diameter (float, optional): The diameter of the spot. Defaults to None.

    Returns:
        tuple: The closest hexagon centroid coordinates (x, y) rounded to the nearest integer.
               Returns -1 if the spot diameter is provided and the distance to the closest
               hexagon centroid is greater than half the spot diameter.
    """
    spot = True if spot_diameter != None else False

    x_ = x // (hexagon_size * 2)
    y_ = y // (hexagon_size * 1.732050807)

    if y_ % 2 == 1:

        # lower_left
        option_1_hexagon = (x_ * 2 * hexagon_size, y_ * 1.732050807 * hexagon_size)

        # lower right
        option_2_hexagon = (
            (x_ + 1) * 2 * hexagon_size,
            y_ * 1.732050807 * hexagon_size,
        )

        # upper middle
        option_3_hexagon = (
            (x_ + 0.5) * 2 * hexagon_size,
            (y_ + 1) * 1.732050807 * hexagon_size,
        )

        # calc distance from each option and return the closest
        distance_1 = math.sqrt(
            (x - option_1_hexagon[0]) ** 2 + (y - option_1_hexagon[1]) ** 2
        )
        distance_2 = math.sqrt(
            (x - option_2_hexagon[0]) ** 2 + (y - option_2_hexagon[1]) ** 2
        )
        distance_3 = math.sqrt(
            (x - option_3_hexagon[0]) ** 2 + (y - option_3_hexagon[1]) ** 2
        )
        closest = [option_1_hexagon, option_2_hexagon, option_3_hexagon][
            np.argmin([distance_1, distance_2, distance_3])
        ]

    else:
        # lower middle
        option_1_hexagon = (
            (x_ + 0.5) * 2 * hexagon_size,
            y_ * (1.732050807 * hexagon_size),
        )

        # upper left
        option_2_hexagon = (
            x_ * 2 * hexagon_size,
            (y_ + 1) * 1.732050807 * hexagon_size,
        )

        # upper right
        option_3_hexagon = (
            (x_ + 1) * 2 * hexagon_size,
            (y_ + 1) * 1.732050807 * hexagon_size,
        )

        # calc distance from each option and return the closest
        distance_1 = math.sqrt(
            (x - option_1_hexagon[0]) ** 2 + (y - option_1_hexagon[1]) ** 2
        )
        distance_2 = math.sqrt(
            (x - option_2_hexagon[0]) ** 2 + (y - option_2_hexagon[1]) ** 2
        )
        distance_3 = math.sqrt(
            (x - option_3_hexagon[0]) ** 2 + (y - option_3_hexagon[1]) ** 2
        )
        closest = [option_1_hexagon, option_2_hexagon, option_3_hexagon][
            np.argmin([distance_1, distance_2, distance_3])
        ]

    closest = (round(closest[0], 0), round(closest[1], 1))
    if spot:
        if math.sqrt((x - closest[0]) ** 2 + (y - closest[1]) ** 2) < spot_diameter / 2:
            return closest
        else:
            return -1
    else:
        return closest


def preprocess_csv(csv_file, batch_size, fieldnames, feature_colname):
    """
    Preprocesses a CSV file by splitting it into smaller batches.

    Args:
        csv_file (str): The path to the CSV file.
        batch_size (int): The number of rows per batch.
        fieldnames (list): The list of field names to include in the batch CSV files.
        feature_colname (str): The name of the column containing feature names.

    Returns:
        tuple: A tuple containing the temporary directory path, the total number of batches created,
               and the sorted unique features.
    """
    unique_features = set()
    tmp_dir = tempfile.mkdtemp(prefix="tmp_hexa")
    print(f"Created temporary directory {tmp_dir}")

    # Open the CSV file based on its format (gzipped or regular)
    if csv_file.endswith(".gz"):
        file_open_fn = gzip.open
        file_open_mode = "rt"
    else:
        file_open_fn = open
        file_open_mode = "r"

    with file_open_fn(csv_file, file_open_mode) as file:
        print("Now creating batches")
        reader = csv.DictReader(file)
        batch_num = 0

        while True:
            batch = list(itertools.islice(reader, batch_size))
            if not batch:
                break  # Exit the loop if batch is empty

            batch_file = os.path.join(tmp_dir, f"batch_{batch_num}.csv")
            with open(batch_file, "w", newline="") as batch_csv:
                writer = csv.writer(batch_csv)
                writer.writerow(
                    fieldnames
                )  # Write header using the provided fieldnames
                writer.writerows(
                    [[row[field] for field in fieldnames] for row in batch]
                )  # Write rows

            unique_features.update(row[feature_colname] for row in batch)
            batch_num += 1
            print(f"Created batch {batch_num}")

    # Convert the set of unique features to a sorted numpy array
    unique_features = np.array(sorted(unique_features))

    print(f"Finished preprocessing. Total batches created: {batch_num}")
    return tmp_dir, batch_num, unique_features


def process_batch(
    batch_file,
    unique_features,
    hexagon_size,
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
):
    """
    Processes a batch CSV file to calculate hexagon counts and cell counts.

    Args:
        batch_file (str): The path to the batch CSV file.
        unique_features (numpy.ndarray): The sorted unique features.
        hexagon_size (float): The size of the hexagon.
        feature_colname (str): The name of the feature column.
        x_colname (str): The name of the x-coordinate column.
        y_colname (str): The name of the y-coordinate column.
        cell_id_colname (str): The name of the cell ID column.
        quality_colname (str, optional): The name of the quality score column. Defaults to None.
        quality_filter (bool, optional): Whether to filter rows based on quality score. Defaults to False.
        count_colname (str, optional): The name of the count column. Defaults to "NA".
        smoothing (bool, optional): Whether to apply smoothing to the counts. Defaults to False.
        quality_per_hexagon (bool, optional): Whether to calculate quality per hexagon. Defaults to False.
        quality_per_probe (bool, optional): Whether to calculate quality per probe. Defaults to False.
        coord_to_um_conversion (float, optional): The conversion factor from coordinates to micrometers. Defaults to 1.
        spot_diameter (float, optional): The diameter of the spot. Defaults to None.

    Returns:
        tuple: A tuple containing the hexagon counts, hexagon cell counts, hexagon quality,
               and probe quality dictionaries.
    """

    # read in file with pandas
    df_batch = pd.read_csv(batch_file)
    
    #adjusting coordinates
    df_batch[x_colname] = (df_batch[x_colname]) * coord_to_um_conversion
    df_batch[y_colname] = (df_batch[y_colname]) * coord_to_um_conversion
    
    #smoothing, generally only for Visium HD or Curio
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
    hexagons = np.array(
        [
            str(closest_hex(x, y, hexagon_size, spot_diameter))
            for x, y in zip(df_batch[x_colname], df_batch[y_colname])
        ]
    )
    df_batch["hexagons"] = hexagons
    # filter out rows where hexagon is -1
    df_batch = df_batch[df_batch["hexagons"] != "-1"]
    df_batch[feature_colname] = df_batch[feature_colname].map(
        {feature: i for i, feature in enumerate(unique_features)}
    )
    # create a dok matrix to store the counts, which is
    counts = (
        np.ones(df_batch.shape[0])
        if count_colname == "NA"
        else df_batch[count_colname].values
    )
    if smoothing != False:
        counts = counts / 4

    df_batch["counts"] = counts


    if quality_per_hexagon == True:
        # create hexagon_quality from df_batch
        hexagon_quality = df_batch.groupby("hexagons")[quality_colname].agg(
            ["mean", "count"]
        )
        print(f"Type of hexagon_quality: {type(hexagon_quality)}")
        print(hexagon_quality)
        if isinstance(hexagon_quality, pd.DataFrame):
            hexagon_quality = hexagon_quality.to_dict(orient="index")
        else:
            print("hexagon_quality is not a DataFrame. Skipping conversion to dictionary.")
            
        

    if quality_per_probe == True:
        # create probe_quality from df_batch
        probe_quality = df_batch.groupby(feature_colname)[quality_colname].agg(
            ["mean", "count"]
        )
        if isinstance(probe_quality, pd.DataFrame):
            probe_quality = probe_quality.to_dict(orient="index")
        else:
            print("probe_quality is not a DataFrame. Skipping conversion to dictionary.")
            print(f"Type of probe_quality: {type(probe_quality)}")
            print(probe_quality)
        
        

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
        hexagon_cell_counts= (
        df_batch[["hexagons", cell_id_colname, "counts"]]
        .groupby(["hexagons", cell_id_colname])
        .aggregate({"counts": "sum"})
        .reset_index())

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
        adata (AnnData): The AnnData object to be written.
        file (str): The file name to be written to. If no extension is provided, '.h5' is appended.

    Returns:
        None
    """

    if ".h5" not in file:
        file = f"{file}.h5"
    if Path(file).exists():
        print(f"File `{file}` already exists. Removing it.")
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
    hexagon_size,
    batch_size=1000000,
    technology="Xenium",
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
    h5_x_colname="x",
    h5_y_colname="y",
    coord_to_um_conversion=1,
    spot_diameter=None,
):
    """
    Processes a CSV file to calculate hexagon counts and cell counts using parallel processing.

    Args:
        csv_file (str): The path to the CSV file.
        hexagon_size (float): The size of the hexagon.
        batch_size (int, optional): The number of rows per batch. Defaults to 1000000.
        technology (str, optional): The technology used. Defaults to "Xenium".
        feature_colname (str, optional): The name of the feature column. Defaults to "feature_name".
        x_colname (str, optional): The name of the x-coordinate column. Defaults to "x_location".
        y_colname (str, optional): The name of the y-coordinate column. Defaults to "y_location".
        cell_id_colname (str, optional): The name of the cell ID column. Defaults to "None".
        quality_colname (str, optional): The name of the quality score column. Defaults to "qv".
        max_workers (int, optional): The maximum number of worker processes to use. Defaults to min(2, multiprocessing.cpu_count()).
        quality_filter (bool, optional): Whether to filter rows based on quality score. Defaults to False.
        count_colname (str, optional): The name of the count column. Defaults to "NA".
        smoothing (bool, optional): Whether to apply smoothing to the counts. Defaults to False.
        quality_per_hexagon (bool, optional): Whether to calculate quality per hexagon. Defaults to False.
        quality_per_probe (bool, optional): Whether to calculate quality per probe. Defaults to False.
        h5_x_colname (str, optional): The name of the x-coordinate column in the h5 file. Defaults to "x".
        h5_y_colname (str, optional): The name of the y-coordinate column in the h5 file. Defaults to "y".
        coord_to_um_conversion (float, optional): The conversion factor from coordinates to micrometers. Defaults to 1.
        spot_diameter (float, optional): The diameter of the spot. Defaults to None.

    Returns:
        tuple: A tuple containing the hexagon counts, hexagon cell counts, hexagon quality, and probe quality dictionaries.
    """

    print(f"Quality filter is set to {quality_filter}")
    print(f"Quality counting per hexagon is set to {quality_per_hexagon}")
    print(f"Quality counting per probe is set to {quality_per_probe}")
    spot = True if spot_diameter != None else False
    if spot:
        print(
            "Visium-like spots are going to be used rather than hexagonal tesselation!!!"
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

    tmp_dir, num_batches, unique_features = preprocess_csv(
        csv_file, batch_size, fieldnames, feature_colname
    )

    hexagon_quality = {}
    probe_quality = {}
    hexagon_counts = pd.DataFrame()
    hexagon_cell_counts = pd.DataFrame()


    batch_files = [os.path.join(tmp_dir, f"batch_{i}.csv") for i in range(num_batches)]
    n_process = min(max_workers, multiprocessing.cpu_count())
    print(f"Processing batches using {n_process} processes")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = [
            executor.submit(
                process_batch,
                batch_file,
                unique_features,
                hexagon_size,
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
            )
            for batch_file in batch_files
        ]

        with tqdm(
            total=len(batch_files), desc="Processing batches", unit="batch"
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
                    [hexagon_cell_counts, batch_hexagon_cell_counts], axis=0)

                    hexagon_cell_counts = (
                    hexagon_cell_counts[["hexagons", cell_id_colname, "counts"]]
                    .groupby(["hexagons", cell_id_colname])
                    .aggregate({"counts": "sum"})
                    .reset_index())

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
                            print(f"Error in trying to add to hexagon_quality")

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
                            print(f"Error in trying to add to probe_quality")
                progress_bar.update(1)

    unique_hexagons = hexagon_counts["hexagons"].unique()
    hexagon_counts["hexagon_id"] = hexagon_counts["hexagons"].map(
        {hexagon: i for i, hexagon in enumerate(unique_hexagons)}
    )
    if cell_id_colname != "None":
        hexagon_cell_counts["hexagon_id"] = hexagon_cell_counts["hexagons"].map(
            {hexagon: i for i, hexagon in enumerate(unique_hexagons)}
        )

    hexagon_counts = scipy.sparse.csr_matrix(
        (
            hexagon_counts["counts"],
            (hexagon_counts["hexagon_id"], hexagon_counts[feature_colname]),
        ),
        shape=(len(hexagon_counts["hexagons"].unique()), len(unique_features)),
    )

    shutil.rmtree(tmp_dir)  # Remove temporary directory and files

    return (
        hexagon_counts,
        unique_hexagons,
        unique_features,
        hexagon_cell_counts,
        hexagon_quality,
        probe_quality,
    )


def create_pseudovisium(
    path,
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
    image_pixels_per_um=1 / 0.2125,
    hexagon_size=100,
    tissue_hires_scalef=0.2,
    pixel_to_micron=False,
    max_workers=min(2, multiprocessing.cpu_count()),
    spot_diameter=None,
):
    """
    Creates a Pseudovisium output directory structure and files.

    Args:
        path (str): The path to create the Pseudovisium output directory.
        hexagon_counts (dict): A dictionary of hexagon counts.
        hexagon_cell_counts (dict): A dictionary of hexagon cell counts.
        hexagon_quality (dict): A dictionary of hexagon quality scores.
        probe_quality (dict): A dictionary of probe quality scores.
        img_file_path (str, optional): The path to the image file. Defaults to None.
        project_name (str, optional): The name of the project. Defaults to "project".
        alignment_matrix_file (str, optional): The path to the alignment matrix file. Defaults to None.
        image_pixels_per_um (float, optional): The number of image pixels per micrometer. Defaults to 1/0.2125.
        hexagon_size (int, optional): The size of the hexagon. Defaults to 100.
        tissue_hires_scalef (float, optional): The scaling factor for the high-resolution tissue image. Defaults to 0.2.
        pixel_to_micron (bool, optional): Whether to convert pixel coordinates to micron coordinates. Defaults to False.
        max_workers (int, optional): The maximum number of worker processes to use. Defaults to min(2, multiprocessing.cpu_count()).
        spot_diameter (float, optional): The diameter of the spot. Defaults to None.
    """
    spot = True if spot_diameter != None else False
    if spot:
        print(
            "Visium-like array structure is being built rather than hexagonal tesselation!!!"
        )

    # to path, create a folder called pseudovisium
    folderpath = path + "/pseudovisium/" + project_name
    # if folderpath exists, delete it
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
    try:
        print("Creating pseudovisium folder in output path:{0}".format(folderpath))

        if os.path.exists(path + "/pseudovisium/"):
            print("Using already existing folder: {0}".format(path + "/pseudovisium/"))
        else:
            os.mkdir(path + "/pseudovisium/")
        os.mkdir(folderpath)
        os.mkdir(folderpath + "/spatial")
    except:
        pass

    ############################################## ##############################################
    # see https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    # https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs

    scalefactors = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "tissue_lowres_scalef": tissue_hires_scalef / 10,
        "fiducial_diameter_fullres": 0,
    }

    if spot:
        scalefactors["spot_diameter_fullres"] = spot_diameter * image_pixels_per_um
    else:
        scalefactors["spot_diameter_fullres"] = 2 * hexagon_size * image_pixels_per_um

    print("Creating scalefactors_json.json file in spatial folder.")
    with open(folderpath + "/spatial/scalefactors_json.json", "w") as f:
        json.dump(scalefactors, f)
    ############################################## ##############################################

    x, y, x_, y_, contain = [], [], [], [], []
    for i, hexagon in enumerate(unique_hexagons):
        # convert hexagon back to tuple from its string form
        hexagon = eval(hexagon)
        x_.append((hexagon[0] + hexagon_size) // (2 * hexagon_size))
        y_.append(hexagon[1] // (1.73205 * hexagon_size))
        x.append(hexagon[0])
        y.append(hexagon[1])
        contain.append(1 if hexagon_counts[i].sum() > hexagon_size else 0)

    ############################################## ##############################################
    barcodes = ["hexagon_{}".format(i) for i in range(1, len(unique_hexagons) + 1)]
    barcodes_table = pd.DataFrame({"barcode": barcodes})
    # save to pseudo visium root
    barcodes_table.to_csv(
        folderpath + "/barcodes.tsv", sep="\t", index=False, header=False
    )

    print("Creating barcodes.tsv.gz file in spatial folder.")
    with open(folderpath + "/barcodes.tsv", "rb") as f_in:
        with gzip.open(folderpath + "/barcodes.tsv.gz", "wb") as f_out:
            f_out.writelines(f_in)
    ############################################## ##############################################
    hexagon_table = pd.DataFrame(
        zip(
            barcodes,
            contain,
            y_,
            x_,
            [int(image_pixels_per_um * a) for a in y],
            [int(image_pixels_per_um * a) for a in x],
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

    print("Creating tissue_positions_list.csv file in spatial folder.")
    hexagon_table.to_csv(
        folderpath + "/spatial/tissue_positions_list.csv", index=False, header=False
    )

    ############################################## ##############################################

    #if hexagon_cell_counts is pandas df
    if hexagon_cell_counts.empty:
        print("No cell information provided. Skipping cell information files.")
    else:
        print("Creating pv_cell_hex.csv file in spatial folder.")
        hexagon_cell_counts = hexagon_cell_counts[[cell_id_colname, "hexagon_id", "counts"]]
        #add 1 to hexagon_ids
        hexagon_cell_counts["hexagon_id"] = hexagon_cell_counts["hexagon_id"] + 1
        #save csv
        hexagon_cell_counts.to_csv(
            folderpath + "/spatial/pv_cell_hex.csv", index=False, header=False
        )


    if hexagon_quality == {}:
        print("No quality information provided. Skipping quality information files.")
    else:
        print("Creating quality_per_hexagon.csv file in spatial folder.")
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
                        f"Error: Unable to find hexagon '{hexagon}' in unique_hexagons."
                    )
                    print(f"Error details: {str(e)}")
                    print(
                        f"Skipping quality measurement for hexagon '{hexagon}' with mean {quality_dict['mean']} and count {quality_dict['count']}."
                    )

    ############################################## ##############################################
    if probe_quality == {}:
        print("No quality information provided. Skipping quality information files.")
    else:
        print("Creating quality_per_probe.csv file in spatial folder.")
        #map back probe_quality indices to unique_features
        #iterate through its keys, and rename it 
        for key in list(probe_quality.keys()):
            probe_quality[unique_features[key]] = probe_quality.pop(key)

        with open(folderpath + "/spatial/quality_per_probe.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for probe, quality_dict in probe_quality.items():
                writer.writerow([probe, quality_dict["mean"], quality_dict["count"]])

    ############################################## ##############################################

    features = unique_features
    # Create a list of rows with repeated features and 'Gene Expression' column
    rows = [[feature, feature, "Gene Expression"] for feature in features]

    print("Creating features.tsv.gz file in spatial folder.")
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

    print("Putting together the matrix.mtx file")

    print(f"Total matrix count: {hexagon_counts.sum()}")
    print(f"Number of unique hexagons: {len(barcodes)}")

    print("Creating matrix.mtx.gz file in spatial folder.")
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

    print("Putting together the filtered_feature_bc_matrix.h5 file")

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
            "Alignment matrix found and will be used to create tissue_hires_image.png and tissue_lowres_image.png files in spatial folder."
        )
        M = pd.read_csv(alignment_matrix_file, header=None, index_col=None).to_numpy()
    else:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Load the H&E image
    if img_file_path:
        print("Image provided at {0}".format(img_file_path))
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
        # if 2, then triple the image to make it thre channels
        if dims == 2:
            image = np.array([image, image, image])
            # change order of axis
            image = np.moveaxis(image, 0, -1)

            resized = np.array([resized, resized, resized])
            # change order of axis
            resized = np.moveaxis(resized, 0, -1)

        print("Creating tissue_hires_image.png file in spatial folder.")
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
        print("No image file provided. Drawing tissue positions on a white background.")
        
        # Extract pixel coordinates from hexagon_table
        pixel_coords = np.array([[int(row*tissue_hires_scalef), int(col*tissue_hires_scalef)] for row, col in zip(hexagon_table["pxl_row_in_fullres"], hexagon_table["pxl_col_in_fullres"])])
        
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
    Reads the necessary files from a Visium HD or Curio folder.

    Args:
        folder (str): The path to the Visium HD or Curio folder.
        technology (str): The technology used, either "Visium_HD" or "Curio".

    Returns:
        tuple or AnnData: A tuple containing the scale factors, tissue positions, and filtered feature-barcode matrix (for Visium HD),
                          or an AnnData object (for Curio).
    """

    if (technology == "Visium_HD") or (technology == "VisiumHD") or (technology == "Visium HD"):
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


def anndata_to_df(
    adata, technology, tissue_pos=None, scalefactors=None, x_col=None, y_col=None
):
    """
    Converts an AnnData object to a DataFrame.

    Args:
        adata (AnnData): The AnnData object containing the counts matrix.
        technology (str): The technology used, either "Visium_HD" or "Curio".
        tissue_pos (DataFrame, optional): The tissue positions DataFrame (for Visium HD). Defaults to None.
        scalefactors (dict, optional): The scale factors dictionary (for Visium HD). Defaults to None.
        x_col (str, optional): The name of the x-coordinate column (for Curio). Defaults to None.
        y_col (str, optional): The name of the y-coordinate column (for Curio). Defaults to None.

    Returns:
        tuple: A tuple containing the converted DataFrame and the image resolution or scale.
    """

    if (technology == "Visium_HD") or (technology == "VisiumHD") or (technology == "Visium HD"):
        # get image resolution of the hires image
        image_resolution = (
            scalefactors["microns_per_pixel"] / scalefactors["tissue_hires_scalef"]
        )
        # change to pixel per micron
        image_resolution = 1 / image_resolution
        # tissue_pos keep barcode, pxl_row_in_fullres, pxl_col_in_fullres
        tissue_pos = tissue_pos[["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"]]
        # Convert the AnnData matrix to a sparse matrix (CSR format)
        X = scipy.sparse.csr_matrix(adata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()

        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame(
            {
                "barcode": adata.obs_names[row_indices],
                "gene": adata.var_names[col_indices],
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
        obs = adata.obs[[x_col, y_col]]
        # rename these cols to x and y
        obs.columns = ["x", "y"]
        scale = np.round(
            min(
                [abs(x1 - x2) for x1 in obs["y"] for x2 in [obs["y"][500]] if x1 != x2]
            ),
            3,
        )  # should be 10um

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
        return df, scale


def visium_hd_curio_to_transcripts(folder, output, technology, x_col=None, y_col=None):
    """
    Converts Visium HD or Curio files to a transcripts CSV file.

    Args:
        folder (str): The path to the Visium HD or Curio folder.
        output (str): The path to save the transcripts CSV file.
        technology (str): The technology used, either "Visium_HD" or "Curio".
        x_col (str, optional): The name of the x-coordinate column (for Curio). Defaults to None.
        y_col (str, optional): The name of the y-coordinate column (for Curio). Defaults to None.

    Returns:
        float: The image resolution (pixels per micrometer) for Visium HD, or the scale for Curio.
    """

    if (technology == "Visium_HD") or (technology == "VisiumHD") or (technology == "Visium HD"):
        scalefactors, tissue_pos, fb_matrix = read_files(folder, technology)
        df, image_resolution = anndata_to_df(
            adata=fb_matrix,
            technology=technology,
            tissue_pos=tissue_pos,
            scalefactors=scalefactors,
        )
        df.to_csv(output, index=False)
        return image_resolution
    elif technology == "Curio":
        adata = read_files(folder, technology)
        df, scale = anndata_to_df(
            adata=adata, technology=technology, x_col=x_col, y_col=y_col
        )
        df.to_csv(output, index=False)
        return scale


######### Main function to generate pseudovisium output ############################################################################################################


def generate_pv(
    csv_file,
    img_file_path=None,
    shift_to_positive=False,
    hexagon_size=100,
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
    visium_hd_folder=None,
    smoothing=False,
    quality_per_hexagon=False,
    quality_per_probe=False,
    h5_x_colname="x",
    h5_y_colname="y",
    coord_to_um_conversion=1,
    spot_diameter=None,
):
    """
    Generates a Pseudovisium output from a CSV file or Visium HD/Curio folder.

    Args:
        csv_file (str): The path to the CSV file.
        img_file_path (str, optional): The path to the image file. Defaults to None.
        hexagon_size (int, optional): The size of the hexagon. Defaults to 100.
        output_path (str, optional): The path to save the Pseudovisium output. Defaults to None.
        batch_size (int, optional): The number of rows per batch. Defaults to 1000000.
        alignment_matrix_file (str, optional): The path to the alignment matrix file. Defaults to None.
        project_name (str, optional): The name of the project. Defaults to 'project'.
        image_pixels_per_um (float, optional): The number of image pixels per micrometer. Defaults to 1.
        tissue_hires_scalef (float, optional): The scaling factor for the high-resolution tissue image. Defaults to 0.2.
        technology (str, optional): The technology used. Defaults to "Xenium".
        feature_colname (str, optional): The name of the feature column. Defaults to "feature_name".
        x_colname (str, optional): The name of the x-coordinate column. Defaults to "x_location".
        y_colname (str, optional): The name of the y-coordinate column. Defaults to "y_location".
        cell_id_colname (str, optional): The name of the cell ID column. Defaults to "None".
        quality_colname (str, optional): The name of the quality score column. Defaults to "qv".
        pixel_to_micron (bool, optional): Whether to convert pixel coordinates to micron coordinates. Defaults to False.
        max_workers (int, optional): The maximum number of worker processes to use. Defaults to min(2, multiprocessing.cpu_count()).
        quality_filter (bool, optional): Whether to filter rows based on quality score. Defaults to False.
        count_colname (str, optional): The name of the count column. Defaults to "NA".
        visium_hd_folder (str, optional): The path to the Visium HD folder. Defaults to None.
        smoothing (float, optional): The smoothing factor. Defaults to False.
        quality_per_hexagon (bool, optional): Whether to calculate quality per hexagon. Defaults to False.
        quality_per_probe (bool, optional): Whether to calculate quality per probe. Defaults to False.
        h5_x_colname (str, optional): The name of the x-coordinate column in the h5 file. Defaults to "x".
        h5_y_colname (str, optional): The name of the y-coordinate column in the h5 file. Defaults to "y".
        coord_to_um_conversion (float, optional): The conversion factor from coordinates to micrometers. Defaults to 1.
        spot_diameter (float, optional): The diameter of the spot. Defaults to None.
    """
    try:
        output = subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip().split('\n')
        version = [x for x in output if 'Pseudovisium' in x]
        date = str(datetime.datetime.now().date())
        print("You are using version: ",version)
        print("Date: ",date)
    except:
        output = subprocess.check_output(['pip3', 'freeze']).decode('utf-8').strip().split('\n')
        version = [x for x in output if 'Pseudovisium' in x]
        date = str(datetime.datetime.now().date())
        print("You are using version: ",version)
        print("Date: ",date)
    

    try:

        start = time.time()

        if (technology == "Visium_HD") or (technology == "VisiumHD") or (technology == "Visium HD"):
            print(
                "Technology is Visium_HD. Generating transcripts.csv file from Visium HD files."
            )
            # Automatically calculating image_pixels_per_um from the scalefactors_json.json file
            image_pixels_per_um = visium_hd_curio_to_transcripts(
                visium_hd_folder, visium_hd_folder + "/transcripts.csv", technology
            )
            csv_file = visium_hd_folder + "/transcripts.csv"
        if technology == "Curio":
            print(
                "Technology is Curio. Generating transcripts.csv file from Curio files."
            )
            smoothing_scale = visium_hd_curio_to_transcripts(
                visium_hd_folder,
                visium_hd_folder + "/transcripts.csv",
                technology,
                x_col=h5_x_colname,
                y_col=h5_y_colname,
            )
            csv_file = visium_hd_folder + "/transcripts.csv"
            print("Smoothing defaults to : {0}".format(smoothing_scale / 4))
            smoothing = smoothing_scale / 4

        if technology == "Xenium":
            print("Technology is Xenium. Going forward with default column names.")
            x_colname = "x_location"
            y_colname = "y_location"
            feature_colname = "feature_name"
            cell_id_colname = "cell_id"
            quality_colname = "qv"
            count_colname = "NA"
            coord_to_um_conversion = 1

        elif technology == "Vizgen":
            print("Technology is Vizgen. Going forward with default column names.")
            x_colname = "global_x"
            y_colname = "global_y"
            feature_colname = "gene"
            # cell_id_colname = "barcode_id"
            count_colname = "NA"
            coord_to_um_conversion = 1

        elif (technology == "Nanostring") or (technology == "CosMx"):
            print("Technology is Nanostring. Going forward with default column names.")
            x_colname = "x_global_px"
            y_colname = "y_global_px"
            feature_colname = "target"
            cell_id_colname = "cell"
            count_colname = "NA"
            coord_to_um_conversion = 0.12028
            # see ref https://smi-public.objects.liquidweb.services/cosmx-wtx/Pancreas-CosMx-ReadMe.html
            # https://nanostring.com/wp-content/uploads/2023/09/SMI-ReadMe-BETA_humanBrainRelease.html
            # Whereas old smi output seems to be 0.18
            # https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html

        elif (technology == "Visium_HD") or (technology == "VisiumHD") or (technology == "Visium HD"):
            print(
                "Technology is Visium_HD. Going forward with pseudovisium processed colnames."
            )
            x_colname = "x"
            y_colname = "y"
            feature_colname = "gene"
            cell_id_colname = "barcode"
            count_colname = "count"
            coord_to_um_conversion = 1

        elif technology == "seqFISH":
            print("Technology is seqFISH. Going forward with default column names.")
            x_colname = "x"
            y_colname = "y"
            feature_colname = "name"
            cell_id_colname = "cell"
            count_colname = "NA"
            coord_to_um_conversion = 1

        elif technology == "Curio":
            print("Technology is Curio. Going forward with default column names.")
            x_colname = "x"
            y_colname = "y"
            feature_colname = "gene"
            cell_id_colname = "barcode"
            count_colname = "count"
            coord_to_um_conversion = 1

        else:
            print("Technology not recognized. Going forward with set column names.")

        (
            hexagon_counts,
            unique_hexagons,
            unique_features,
            hexagon_cell_counts,
            hexagon_quality,
            probe_quality,
        ) = process_csv_file(
            csv_file,
            hexagon_size,
            batch_size,
            technology,
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
            h5_x_colname=h5_x_colname,
            h5_y_colname=h5_y_colname,
            coord_to_um_conversion=coord_to_um_conversion,
            spot_diameter=spot_diameter,
        )

        # Create Pseudovisium output
        create_pseudovisium(
            path=output_path,
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
            hexagon_size=hexagon_size,
            tissue_hires_scalef=tissue_hires_scalef,
            pixel_to_micron=pixel_to_micron,
            max_workers=max_workers,
            spot_diameter=spot_diameter,
        )

        # save all arguments in a json file called arguments.json
        print("Creating arguments.json file in output path.")
        arguments = {
            "csv_file": csv_file,
            "img_file_path": img_file_path,
            "hexagon_size": hexagon_size,
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
            "visium_hd_folder": visium_hd_folder,
            "h5_x_colname": h5_x_colname,
            "h5_y_colname": h5_y_colname,
            "coord_to_um_conversion": coord_to_um_conversion,
            "spot_diameter": spot_diameter,
        }

        with open(
            output_path + "/pseudovisium/" + project_name + "/arguments.json", "w"
        ) as f:
            json.dump(arguments, f)

        end = time.time()
        print(f"Time taken: {end - start} seconds")
    finally:
        delete_temporary_files()


def main():
    """
    The main function that parses command-line arguments and calls the generate_pv function.
    """
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument(
        "--csv_file", "-c", type=str, help="CSV file path", default=None
    )
    parser.add_argument(
        "--output_path", "-o", type=str, help="Output path", default="."
    )
    parser.add_argument(
        "--hexagon_size", "-hs", type=float, help="Hexagon size", default=100.0
    )
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
        "--visium_hd_folder", "-vhf", type=str, help="Visium HD folder", default=None
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
    parser.add_argument(
        "-help", action="store_true", help="Print out script purpose and parameters"
    )
    args = parser.parse_args()

    if args.help:
        print(
            "This is Pseudovisium, a software that compresses imaging-based spatial transcriptomics files using hexagonal binning of the data."
        )
        parser.print_help()
        sys.exit(0)

    generate_pv(
        csv_file=args.csv_file,
        img_file_path=args.img_file_path,
        shift_to_positive=args.shift_to_positive,
        hexagon_size=args.hexagon_size,
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
        visium_hd_folder=args.visium_hd_folder,
        spot_diameter=args.spot_diameter,
    )

    print("Pseudovisium output generated successfully.")


if __name__ == "__main__":
    main()
