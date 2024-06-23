import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import gzip
import cv2
import scanpy as sc
import scipy.io
import h5py
from pathlib import Path
import subprocess
import datetime


def from_h5_to_files(fold):
    """
    Convert filtered_feature_bc_matrix.h5 file to features.tsv, barcodes.tsv, and matrix.mtx files.

    Args:
        fold (str): Path to the folder containing the filtered_feature_bc_matrix.h5 file.

    Returns:
        None
    """
    # look for h5 ending file
    h5file = [f for f in os.listdir(fold) if f.endswith(".h5")][0]
    ad = sc.read_10x_h5(fold + h5file)
    ad.var_names_make_unique()
    features = pd.DataFrame(ad.var.index)
    features[1] = features[0]
    features[2] = "Gene Expression"
    features.to_csv(
        os.path.join(fold, "features.tsv"), sep="\t", index=False, header=False
    )
    pd.DataFrame(ad.obs.index).to_csv(
        os.path.join(fold, "barcodes.tsv"), sep="\t", index=False, header=False
    )
    scipy.io.mmwrite(os.path.join(fold, "matrix.mtx"), ad.X.T)


def load_data(folder):
    """
    Load data from a folder containing Visium/Pseudovisium output files.

    Args:
        folder (str): Path to the folder containing Visium/Pseudovisium output files.

    Returns:
        tuple: A tuple containing:
            - dataset_name (str): Name of the dataset.
            - data (dict): A dictionary with loaded data including matrix, tissue positions, features, barcodes, image, scalefactors, and data source.
    """
    dataset_name = folder.split("/")[-2]
    print(f"Loading in {dataset_name}")

    barcodes_file = folder + "/barcodes.tsv"
    features_file = folder + "/features.tsv"
    matrix_file = folder + "/matrix.mtx"

    if (
        not os.path.exists(barcodes_file)
        or not os.path.exists(features_file)
        or not os.path.exists(matrix_file)
    ):
        print(
            f"Missing files in {folder}. Attempting to generate from filtered_feature_bc_matrix.h5."
        )
        from_h5_to_files(folder)

    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")
    barcodes.columns = ["Barcode_ID"]
    barcodes["Barcode_ID"] = [
        f"{barcode}_{dataset_name}" for barcode in barcodes["Barcode_ID"]
    ]
    barcodes["index_col"] = barcodes.index + 1

    tissue_positions_list = pd.read_csv(
        folder + "/spatial/tissue_positions_list.csv", header=None
    )
    tissue_positions_list.columns = [
        "barcode",
        "in_tissue",
        "tissue_col",
        "tissue_row",
        "x",
        "y",
    ]

    scalefactors = json.load(open(folder + "/spatial/scalefactors_json.json"))
    # check if there is an arguments.json file in the folder if yes it is pv if not it is v
    data_source = "pv" if scalefactors["fiducial_diameter_fullres"] == 0 else "v"

    # make sure all coordinates are positive
    # check if either tissue_col, tissue_row, x or y is negative, then find the minimum and add the absolute value to all
    if data_source == "pv":
        if tissue_positions_list["tissue_col"].min() < 0:
            tissue_positions_list["tissue_col"] = tissue_positions_list[
                "tissue_col"
            ] + abs(tissue_positions_list["tissue_col"].min())
        if tissue_positions_list["tissue_row"].min() < 0:
            tissue_positions_list["tissue_row"] = tissue_positions_list[
                "tissue_row"
            ] + abs(tissue_positions_list["tissue_row"].min())
        if tissue_positions_list["x"].min() < 0:
            tissue_positions_list["x"] = tissue_positions_list["x"] + abs(
                tissue_positions_list["x"].min()
            )
        if tissue_positions_list["y"].min() < 0:
            tissue_positions_list["y"] = tissue_positions_list["y"] + abs(
                tissue_positions_list["y"].min()
            )

    tissue_positions_list["barcode"] = [
        f"{barcode}_{dataset_name}" for barcode in tissue_positions_list["barcode"]
    ]
    tissue_positions_list["dataset"] = dataset_name
    barcode_indices = []
    for barcode in tissue_positions_list["barcode"]:
        try:
            barcode_indices.append(
                barcodes[barcodes["Barcode_ID"] == barcode].index_col.values[0]
            )
        except:
            barcode_indices.append(-1)
            pass
    tissue_positions_list["barcode_id"] = barcode_indices

    features = pd.read_csv(features_file, header=None, sep="\t")
    if features.shape[1] == 3:
        features.columns = ["Gene_ID", "Gene_Name", "Type"]
    elif features.shape[1] == 1:
        features.columns = ["Gene_ID"]
    features["index_col"] = features.index + 1

    matrix = pd.read_csv(matrix_file, header=3, sep=" ")
    matrix.columns = ["Gene_ID", "Barcode_ID", "Counts"]

    # make all gene ids uppercase for simplicity
    features["Gene_ID"] = features["Gene_ID"].str.upper()

    if data_source == "v":
        # get correction factor
        # find the fourth and fifth value in tissue_positions_list["y"]
        spot_dist = (
            tissue_positions_list["y"].values[3] - tissue_positions_list["y"].values[4]
        )
        real_dist = 50
        micron_per_pixel = real_dist / abs(spot_dist)
        tissue_positions_list["x"] = tissue_positions_list["x"] * micron_per_pixel
        tissue_positions_list["y"] = tissue_positions_list["y"] * micron_per_pixel
    elif data_source == "pv":
        # get correction factor
        spot_diam = scalefactors["spot_diameter_fullres"]
        real_diam = scalefactors["hexagon_diameter"]
        micron_per_pixel = real_diam / spot_diam
        tissue_positions_list["x"] = tissue_positions_list["x"] * micron_per_pixel
        tissue_positions_list["y"] = tissue_positions_list["y"] * micron_per_pixel

    image_exists = os.path.exists(folder + "/spatial/tissue_hires_image.png")
    if image_exists:
        image = plt.imread(folder + "/spatial/tissue_hires_image.png")
    else:
        image = np.zeros(
            (
                int(tissue_positions_list["x"].max()),
                int(tissue_positions_list["y"].max()),
            )
        )

    data = {
        "matrix": matrix,
        "tissue_positions_list": tissue_positions_list,
        "features": features,
        "barcodes": barcodes,
        "image": image,
        "scalefactors": scalefactors,
        "data_source": data_source,
    }
    return dataset_name, data


def merge_data(folders, pv_format=False):
    """
    Merge data from multiple Visium output folders.

    Args:
        folders (list): List of folder paths containing Visium output files.
        pv_format (bool, optional): Indicate if input is in Pseudovisium format. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - dataset_names (list): List of dataset names.
            - nested_dict (dict): Nested dictionary containing data for each dataset.
            - features_df (pd.DataFrame): DataFrame containing features from all datasets.
            - barcodes_df (pd.DataFrame): DataFrame containing barcodes from all datasets.
            - max_x_range (float): Maximum x-range across all datasets.
            - max_y_range (float): Maximum y-range across all datasets.
            - max_col_range (float): Maximum column range across all datasets.
            - max_row_range (float): Maximum row range across all datasets.
            - scalefactor_hires (float): Minimum high-resolution scale factor across all datasets.
            - scalefactors_for_save (dict): Scale factors for saving.
            - resize_factors (list): List of resize factors for each dataset.
    """

    dataset_names = []
    nested_dict = {}
    features_df = pd.DataFrame()
    barcodes_df = pd.DataFrame()

    scalefactor_hires_vals = []
    for folder in folders:
        scalefactors = json.load(open(folder + "/spatial/scalefactors_json.json"))
        data_source = "pv" if scalefactors["fiducial_diameter_fullres"] == 0 else "v"
        real_spot_diam_pixel = scalefactors["spot_diameter_fullres"]
        if data_source == "v":
            tissue_positions_list = pd.read_csv(
            folder + "/spatial/tissue_positions_list.csv", header=None
            )
            tissue_positions_list.columns = [
                "barcode",
                "in_tissue",
                "tissue_col",
                "tissue_row",
                "x",
                "y",
            ]
            spot_dist = (
                tissue_positions_list["y"].values[3]
                - tissue_positions_list["y"].values[4]
            )
            real_dist = 50
            micron_per_pixel = real_dist / abs(spot_dist)
            real_spot_diam_pixel = scalefactors["spot_diameter_fullres"]
            scalefactors["tissue_hires_scalef"] = (
                scalefactors["tissue_hires_scalef"] / micron_per_pixel
            )
        elif data_source == "pv":
            spot_diam = scalefactors["spot_diameter_fullres"]
            real_diam = scalefactors["hexagon_diameter"]
            micron_per_pixel = real_diam / spot_diam
            real_spot_diam_pixel = spot_diam
            scalefactors["tissue_hires_scalef"] = (
                scalefactors["tissue_hires_scalef"] / micron_per_pixel
            )

        scalefactor_hires_vals.append(scalefactors["tissue_hires_scalef"])

    min_scalefactor = min(scalefactor_hires_vals)
    resize_factors = []
    for scalefactor_hires in scalefactor_hires_vals:
        resize_factors.append(min_scalefactor / scalefactor_hires)

    scalefactor_hires = min_scalefactor
    min_scalefactor_folder = folders[scalefactor_hires_vals.index(min_scalefactor)]
    scalefactors_for_save = json.load(
        open(min_scalefactor_folder + "/spatial/scalefactors_json.json")
    )
    scalefactors_for_save["tissue_hires_scalef"] = scalefactor_hires
    scalefactors_for_save["spot_diameter_fullres"] = real_spot_diam_pixel

    max_x_range = 0
    max_y_range = 0
    max_col_dims = 0
    max_row_dims = 0

    for folder, i in zip(folders, range(len(folders))):
        dataset_name, data = load_data(folder)
        dataset_names.append(dataset_name)
        nested_dict[dataset_name] = data

        # adjust image size
        image = data["image"]
        width = int(image.shape[1] * resize_factors[i])
        height = int(image.shape[0] * resize_factors[i])
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        nested_dict[dataset_name]["image"] = image

        barcodes_df_to_add = data["barcodes"].copy()
        barcodes_df_to_add["dataset"] = dataset_name
        barcodes_df = pd.concat([barcodes_df, barcodes_df_to_add])

        features_df_to_add = data["features"].copy()
        features_df_to_add["dataset"] = dataset_name
        features_df = pd.concat([features_df, features_df_to_add])

        if pv_format:
            pv_cell_hex = pd.read_csv(folder + "/spatial/pv_cell_hex.csv", header=None)
            pv_cell_hex.columns = ["Cell_ID", "Hexagon_ID", "Count"]
            pv_cell_hex["Cell_ID"] = [
                f"{cell_id}_{dataset_name}" for cell_id in pv_cell_hex["Cell_ID"]
            ]
            nested_dict[dataset_name]["pv_cell_hex"] = pv_cell_hex
            arguments = json.load(open(folder + "arguments.json"))
            nested_dict[dataset_name]["arguments"] = arguments

        tissue_positions_list = data["tissue_positions_list"]
        max_x_range = max(max_x_range, tissue_positions_list["x"].max()) * 1.00001
        max_y_range = max(max_y_range, tissue_positions_list["y"].max()) * 1.00001
        max_col_range = (
            max(max_col_dims, tissue_positions_list["tissue_col"].max()) * 1.00001
        )
        max_row_range = (
            max(max_row_dims, tissue_positions_list["tissue_row"].max()) * 1.00001
        )

    return (
        dataset_names,
        nested_dict,
        features_df,
        barcodes_df,
        max_x_range,
        max_y_range,
        max_col_range,
        max_row_range,
        scalefactor_hires,
        scalefactors_for_save,
        resize_factors,
    )


def consensus_features(features_df, only_common):
    """
    Create a consensus index for features across all datasets.

    Args:
        features_df (pd.DataFrame): DataFrame containing features from all datasets.
        only_common (bool): If True, only keep features present in all datasets.

    Returns:
        pd.DataFrame: Updated features DataFrame with consensus index.
    """

    features_df_all = features_df.copy()
    # save to csv
    features_df_all.to_csv("features_df_all1.csv")

    if only_common:
        common_genes = []
        for gene_id in features_df_all["Gene_ID"].unique():
            gene_datasets = features_df_all[features_df_all["Gene_ID"] == gene_id][
                "dataset"
            ].unique()
            if len(gene_datasets) == features_df_all["dataset"].nunique():
                common_genes.append(gene_id)

        features_df_all = features_df_all[features_df_all["Gene_ID"].isin(common_genes)]
        # Print number of such genes
        print(f"Number of genes present in all datasets: {len(common_genes)}")

    unique_gene_ids = features_df_all["Gene_ID"].unique()

    features_df_all["consensus_index"] = features_df_all["Gene_ID"].apply(
        lambda x: np.where(unique_gene_ids == x)[0][0] + 1
    )
    features_df_all.to_csv("features_df_all2.csv")
    return features_df_all


def adjust_matrix_barcodes(
    nested_dict, features_df_all, pv_format=False, only_common=True
):
    """
    Adjust matrix and barcode IDs in the nested dictionary.

    Args:
        nested_dict (dict): Nested dictionary containing data for each dataset.
        features_df_all (pd.DataFrame): DataFrame containing features with consensus index.
        pv_format (bool, optional): Indicate if input is in Pseudovisium format. Defaults to False.
        only_common (bool, optional): If True, only keep features present in all datasets. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - nested_dict (dict): Updated nested dictionary with adjusted matrix and barcode IDs.
            - n_barcodes_before (int): Total number of barcodes across all datasets.
    """
    n_barcodes_before = 0
    for dataset_name in nested_dict.keys():
        print(dataset_name)
        matrix = nested_dict[dataset_name]["matrix"]
        matrix.to_csv("matrix1" + dataset_name + ".csv")
        features_all = features_df_all[features_df_all["dataset"] == dataset_name]
        matrix["new_gene_id"] = matrix["Gene_ID"]
        if only_common:
            all_features_indices = features_all["index_col"].unique()
            matrix = matrix[matrix["Gene_ID"].isin(all_features_indices)]

            for gene_id in matrix["Gene_ID"].unique():
                if gene_id in features_all["index_col"].values:
                    original_index, consensus_index = features_all[
                        features_all["index_col"] == gene_id
                    ][["index_col", "consensus_index"]].values[0]
                    matrix.loc[matrix["Gene_ID"] == gene_id, "new_gene_id"] = (
                        consensus_index
                    )
                else:
                    matrix = matrix[matrix["Gene_ID"] != gene_id]
        else:
            for gene_id in matrix["Gene_ID"].unique():
                original_index, consensus_index = features_all[
                    features_all["index_col"] == gene_id
                ][["index_col", "consensus_index"]].values[0]
                matrix.loc[matrix["Gene_ID"] == gene_id, "new_gene_id"] = (
                    consensus_index
                )

        matrix = matrix[["Barcode_ID", "new_gene_id", "Counts"]]
        matrix.columns = ["Barcode_ID", "Gene_ID", "Counts"]

        matrix["Barcode_ID"] = matrix["Barcode_ID"] + n_barcodes_before
        nested_dict[dataset_name]["matrix"] = matrix
        if pv_format:
            pv_cell_hex = nested_dict[dataset_name]["pv_cell_hex"]
            pv_cell_hex["Hexagon_ID"] = pv_cell_hex["Hexagon_ID"] + n_barcodes_before
            nested_dict[dataset_name]["pv_cell_hex"] = pv_cell_hex
        matrix.to_csv("matrix2" + dataset_name + ".csv")
        n_barcodes_before += len(nested_dict[dataset_name]["barcodes"])
        print("n_barcodes_before", n_barcodes_before)
    return nested_dict, n_barcodes_before


def stitch_images(
    nested_dict,
    grid,
    max_x_range,
    max_y_range,
    max_col_range,
    max_row_range,
    scalefactor_hires,
):
    """
    Stitch images from multiple datasets into a single large image.

    Args:
        nested_dict (dict): Nested dictionary containing data for each dataset.
        grid (list): Grid dimensions for stitching images.
        max_x_range (float): Maximum x-range of the stitched image.
        max_y_range (float): Maximum y-range of the stitched image.
        max_col_range (float): Maximum column range of the stitched image.
        max_row_range (float): Maximum row range of the stitched image.
        scalefactor_hires (float): High-resolution scale factor.

    Returns:
        tuple: A tuple containing:
            - new_tissue_positions_list (pd.DataFrame): DataFrame with updated tissue positions.
            - stitched_image (np.ndarray): Numpy array of the stitched image.
    """

    new_tissue_positions_list = pd.DataFrame(
        columns=["barcode", "in_tissue", "tissue_col", "tissue_row", "x", "y"]
    )
    stitched_image = np.zeros(
        (
            int(max_x_range * scalefactor_hires * grid[0]),
            int(max_y_range * scalefactor_hires * grid[1]),
            3,
        )
    )
    number_of_datasets = len(nested_dict.keys())
    for i in range(grid[0]):
        for j in range(grid[1]):

            dataset_idx = i * grid[1] + j
            if dataset_idx >= number_of_datasets:
                break
            print(dataset_idx)
            dataset_names = list(nested_dict.keys())
            tissue_positions_list_to_add = nested_dict[dataset_names[dataset_idx]][
                "tissue_positions_list"
            ].copy()
            tissue_positions_list_to_add["x"] = (
                tissue_positions_list_to_add["x"] + i * max_x_range
            )
            tissue_positions_list_to_add["y"] = (
                tissue_positions_list_to_add["y"] + j * max_y_range
            )
            tissue_positions_list_to_add["tissue_col"] = (
                tissue_positions_list_to_add["tissue_col"] + i * max_col_range
            )
            tissue_positions_list_to_add["tissue_row"] = (
                tissue_positions_list_to_add["tissue_row"] + j * max_row_range
            )
            new_tissue_positions_list = pd.concat(
                [new_tissue_positions_list, tissue_positions_list_to_add]
            )
            new_tissue_positions_list.reset_index(drop=True, inplace=True)

            image_to_add = nested_dict[dataset_names[dataset_idx]]["image"]
            if len(image_to_add.shape) == 2:
                image_to_add = np.stack((image_to_add,) * 3, axis=-1)
            image_to_add = np.pad(
                image_to_add,
                (
                    (
                        0,
                        max(
                            0,
                            int(max_x_range * scalefactor_hires)
                            - image_to_add.shape[0],
                        ),
                    ),
                    (
                        0,
                        max(
                            0,
                            int(max_y_range * scalefactor_hires)
                            - image_to_add.shape[1],
                        ),
                    ),
                    (0, 0),
                ),
                constant_values=1,
            )
            image_to_add = image_to_add[
                0 : int(max_x_range * scalefactor_hires),
                0 : int(max_y_range * scalefactor_hires),
            ]
            image_to_add = np.pad(
                image_to_add,
                (
                    (0, int(max_x_range * scalefactor_hires) - image_to_add.shape[0]),
                    (0, int(max_y_range * scalefactor_hires) - image_to_add.shape[1]),
                    (0, 0),
                ),
                constant_values=1,
            )
            stitched_image[
                i
                * int(max_x_range * scalefactor_hires) : (i + 1)
                * int(max_x_range * scalefactor_hires),
                j
                * int(max_y_range * scalefactor_hires) : (j + 1)
                * int(max_y_range * scalefactor_hires),
                :,
            ] = image_to_add

    new_tissue_positions_list.drop(["dataset", "barcode_id"], axis=1, inplace=True)
    # round "tissue_col" and "tissue_row" to integers
    new_tissue_positions_list["tissue_col"] = new_tissue_positions_list[
        "tissue_col"
    ].astype(int)
    return new_tissue_positions_list, stitched_image


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


def save_output(
    folderpath,
    scalefactors_for_save,
    barcodes_df_all,
    new_tissue_positions_list,
    pv_format,
    nested_dict,
    dataset_names,
    features_df_all,
    stitched_image,
):
    """
    Save the merged output files.

    Args:
        folderpath (str): Path to the output folder.
        scalefactors_for_save (dict): Scale factors for saving.
        barcodes_df_all (pd.DataFrame): DataFrame containing all barcodes.
        new_tissue_positions_list (pd.DataFrame): DataFrame containing new tissue positions.
        pv_format (bool): Indicate if input is in Pseudovisium format.
        nested_dict (dict): Nested dictionary containing data for each dataset.
        dataset_names (list): List of dataset names.
        features_df_all (pd.DataFrame): DataFrame containing all features.
        stitched_image (np.ndarray): Stitched image array.

    Returns:
        None
    """

    if not os.path.exists(folderpath):
        os.makedirs(folderpath + "/spatial/")

    with open(folderpath + "/spatial/scalefactors_json.json", "w") as f:
        json.dump(scalefactors_for_save, f)

    barcodes_df_all.drop("index_col", axis=1, inplace=True)
    barcodes_df_all.to_csv(
        folderpath + "/barcodes.tsv", sep="\t", index=False, header=False
    )
    print("Generating barcodes.tsv.gz")
    with open(folderpath + "/barcodes.tsv", "rb") as f_in:
        with gzip.open(folderpath + "/barcodes.tsv.gz", "wb") as f_out:
            f_out.writelines(f_in)

    new_tissue_positions_list.to_csv(
        folderpath + "/spatial/tissue_positions_list.csv", index=False, header=False
    )

    if pv_format:
        pv_cell_hex = pd.concat(
            [nested_dict[dataset_name]["pv_cell_hex"] for dataset_name in dataset_names]
        )
        print("Generating pv_cell_hex.csv")
        pv_cell_hex.to_csv(
            folderpath + "/spatial/pv_cell_hex.csv", index=False, header=False
        )

    # remove duplicates
    features_df_all = features_df_all.sort_values("consensus_index").drop_duplicates(
        subset=["Gene_ID"]
    )
    features_df_all["Gene_Name"] = features_df_all["Gene_ID"]
    features_df_all["Type"] = "Gene expression"
    features_df_all = features_df_all[["Gene_ID", "Gene_Name", "Type"]]

    features_df_all.to_csv(
        folderpath + "/features.tsv", sep="\t", index=False, header=False
    )

    with open(folderpath + "/features.tsv", "rb") as f_in, gzip.open(
        folderpath + "/features.tsv.gz", "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)

    matrix_all = pd.concat(
        [nested_dict[dataset_name]["matrix"] for dataset_name in dataset_names]
    )
    # sort Counts
    matrix_all = matrix_all.sort_values("Counts", ascending=False)
    # collapse "Barcode_ID","Gene_ID" duplicates
    matrix_all = (
        matrix_all.groupby(["Gene_ID", "Barcode_ID"])
        .agg({"Counts": "sum"})
        .reset_index()
    )

    print("Generating matrix.mtx.gz")
    metadata = f'%%MatrixMarket matrix coordinate integer general\n%metadata_json: {{"software_version": "Pseudovisium", "format_version": 1}}\n{len(features_df_all)} {barcodes_df_all.shape[0]} {len(matrix_all)}'
    # reorder columns in matrix_all
    print("matrix_head before writing out", matrix_all.head(5))
    matrix_all.to_csv(folderpath + "/matrix.mtx", index=False, header=False, sep=" ")
    with open(folderpath + "/matrix.mtx", "r") as original:
        data = original.read()
    with open(folderpath + "/matrix.mtx", "w") as modified:
        modified.write(metadata + "\n" + data)
    with open(folderpath + "/matrix.mtx", "rb") as f_in, gzip.open(
        folderpath + "/matrix.mtx.gz", "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)

    # create h5 file
    print("Generating filtered_feature_bc_matrix.h5")
    # from matrix_all subtract 1 from barcode_id and gene_id then map to the string barcode_id and gene_id
    matrix_all["Barcode_ID"] = matrix_all["Barcode_ID"] - 1
    matrix_all["Gene_ID"] = matrix_all["Gene_ID"] - 1
    # with lambda function map the barcode_id and gene_id to the string barcode_id and gene_id
    matrix_all["Barcode_ID"] = matrix_all["Barcode_ID"].apply(
        lambda x: barcodes_df_all["Barcode_ID"].iloc[x]
    )
    matrix_all["Gene_ID"] = matrix_all["Gene_ID"].apply(
        lambda x: features_df_all["Gene_ID"].iloc[x]
    )

    matrix_all_pivoted = matrix_all.pivot(
        index="Gene_ID", columns="Barcode_ID", values="Counts"
    ).fillna(0)
    matrix_sparse = scipy.sparse.csr_matrix(matrix_all_pivoted.values).T
    print(matrix_sparse)
    adata = sc.AnnData(
        X=matrix_sparse,
        obs=pd.DataFrame(index=matrix_all_pivoted.columns),
        var=pd.DataFrame(index=matrix_all_pivoted.index),
    )

    write_10X_h5(adata, folderpath + "/filtered_feature_bc_matrix.h5")

    print("Generating tissue_hires_image.png")
    plt.imsave(folderpath + "/spatial/tissue_hires_image.png", stitched_image)
    lowres_image = cv2.resize(
        stitched_image,
        (int(stitched_image.shape[1] / 5), int(stitched_image.shape[0] / 5)),
    )
    # load new scalefactors back in
    scalefactors = json.load(open(folderpath + "/spatial/scalefactors_json.json"))
    scalefactors["tissue_lowres_scalef"] = scalefactors["tissue_hires_scalef"] / 5
    with open(folderpath + "/spatial/scalefactors_json.json", "w") as f:
        json.dump(scalefactors, f)

    print("Generating tissue_lowres_image.png")
    plt.imsave(folderpath + "/spatial/tissue_lowres_image.png", lowres_image)


def merge_visium(folders, output_path, project_name, pv_format=False, only_common=True):
    """
    Merge Pseudovisium/Visium output from multiple folders.

    Args:
        folders (list): List of folder paths containing Pseudovisium/Visium output files.
        output_path (str): Path to the output folder.
        project_name (str): Project name for output.
        pv_format (bool, optional): Indicate if input is in Pseudovisium format. Defaults to False.
        only_common (bool, optional): If True, only keep features present in all datasets. Defaults to True.

    Returns:
        None
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
        print("You are using version: ", version)
        print("Date: ", date)
    except:
        output = (
            subprocess.check_output(["pip3", "freeze"])
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        version = [x for x in output if "Pseudovisium" in x]
        date = str(datetime.datetime.now().date())
        print("You are using version: ", version)
        print("Date: ", date)

    (
        dataset_names,
        nested_dict,
        features_df,
        barcodes_df,
        max_x_range,
        max_y_range,
        max_col_range,
        max_row_range,
        scalefactor_hires,
        scalefactors_for_save,
        resize_factors,
    ) = merge_data(folders, pv_format)
    n_rows = max(3, int(np.sqrt(len(folders))))
    n_cols = int(np.ceil(len(folders) / n_rows))
    grid = [n_rows, n_cols]
    features_df_all = consensus_features(features_df, only_common)
    nested_dict, n_barcodes_before = adjust_matrix_barcodes(
        nested_dict, features_df_all, pv_format, only_common
    )
    barcodes_df_all = barcodes_df.copy()
    barcodes_df_all.reset_index(drop=True, inplace=True)
    barcodes_df_all.drop("dataset", axis=1, inplace=True)
    new_tissue_positions_list, stitched_image = stitch_images(
        nested_dict,
        grid,
        max_x_range,
        max_y_range,
        max_col_range,
        max_row_range,
        scalefactor_hires,
    )
    folderpath = output_path + "/pseudovisium/" + project_name
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
    save_output(
        folderpath,
        scalefactors_for_save,
        barcodes_df_all,
        new_tissue_positions_list,
        pv_format,
        nested_dict,
        dataset_names,
        features_df_all,
        stitched_image,
    )


def main():
    """
    Main function to parse command-line arguments and run the merging process.

    This function sets up the argument parser, processes the input arguments,
    and calls the merge_visium function with the provided parameters.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description="Merge Pseudovisium/Visium format files."
    )
    parser.add_argument(
        "--folders",
        "-f",
        nargs="+",
        help="List of folders containing Pseudovisium/Visium output",
        required=True,
    )
    parser.add_argument(
        "--output_path", "-o", default=os.getcwd(), help="Output folder path"
    )
    parser.add_argument(
        "--project_name", "-p", default="visium_merged", help="Project name for output"
    )
    parser.add_argument(
        "--pv_format",
        "-pvf",
        action="store_true",
        help="Indicate if input is in Pseudovisium format",
    )
    parser.add_argument(
        "--only_common",
        "-oc",
        action="store_true",
        help="Only keep genes present in all datasets",
    )
    args = parser.parse_args()

    folders = [
        folder + "/" if not folder.endswith("/") else folder for folder in args.folders
    ]
    merge_visium(
        folders, args.output_path, args.project_name, args.pv_format, args.only_common
    )


if __name__ == "__main__":
    main()
