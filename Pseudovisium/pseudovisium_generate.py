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

def closest_hex(x,y,hexagon_size):
    """
    closest_hex(x, y, hexagon_size)
    Calculates the closest hexagon centroid to the given (x, y) coordinates.

    Args:
    x (float): The x-coordinate.
    y (float): The y-coordinate.
    hexagon_size (float): The size of the hexagon.

    Returns:
        tuple: The closest hexagon centroid coordinates (x, y) rounded to the nearest integer.
    """

    x_ = x // (hexagon_size*2)
    y_= y // (hexagon_size*1.732050807)


    if y_ % 2 == 1:

        #lower_left
        option_1_hexagon = (x_*2*hexagon_size,y_*1.732050807*hexagon_size)

        #lower right
        option_2_hexagon = ((x_+1)*2*hexagon_size,y_*1.732050807*hexagon_size)

        #upper middle
        option_3_hexagon = ((x_+0.5)*2*hexagon_size,(y_+1)*1.732050807*hexagon_size)

        #calc distance from each option and return the closest
        distance_1 = math.sqrt((x - option_1_hexagon[0])**2 + (y - option_1_hexagon[1])**2)
        distance_2 = math.sqrt((x - option_2_hexagon[0])**2 + (y - option_2_hexagon[1])**2)
        distance_3 = math.sqrt((x - option_3_hexagon[0])**2 + (y - option_3_hexagon[1])**2)
        closest = [option_1_hexagon,option_2_hexagon,option_3_hexagon][np.argmin([distance_1,distance_2,distance_3])]

    else:
            #lower middle 
        option_1_hexagon = ((x_+0.5)*2*hexagon_size,y_*(1.732050807*hexagon_size))

            #upper left
        option_2_hexagon = (x_*2*hexagon_size,(y_+1)*1.732050807*hexagon_size)

            #upper right
        option_3_hexagon = ((x_+1)*2*hexagon_size,(y_+1)*1.732050807*hexagon_size)

        #calc distance from each option and return the closest
        distance_1 = math.sqrt((x - option_1_hexagon[0])**2 + (y - option_1_hexagon[1])**2)
        distance_2 = math.sqrt((x - option_2_hexagon[0])**2 + (y - option_2_hexagon[1])**2)
        distance_3 = math.sqrt((x - option_3_hexagon[0])**2 + (y - option_3_hexagon[1])**2)
        closest = [option_1_hexagon,option_2_hexagon,option_3_hexagon][np.argmin([distance_1,distance_2,distance_3])]

    closest = (round(closest[0], 0), round(closest[1], 1))
    return closest




def preprocess_csv(csv_file, batch_size, fieldnames):
    """
    preprocess_csv(csv_file, batch_size, fieldnames)
    Preprocesses a CSV file by splitting it into smaller batches.

    Args:
        csv_file (str): The path to the CSV file.
        batch_size (int): The number of rows per batch.
        fieldnames (list): The list of field names to include in the batch CSV files.

    Returns:
        tuple: A tuple containing the temporary directory path and the total number of batches created.

    """


    tmp_dir = tempfile.mkdtemp(prefix="tmp_hexa")
    print(f"Created temporary directory {tmp_dir}")
    #if csv_file is .gz, then open it with gzip
    if csv_file.endswith('.gz'):
        with gzip.open(csv_file, 'rt') as file:
            print("Now creating batches")
            reader = csv.DictReader(file)
            header = next(reader)  # Read the header row
            batch_num = 0

            while True:
                batch = list(itertools.islice(reader, batch_size))
                if not batch:
                    break  # Exit the loop if batch is empty

                batch_file = os.path.join(tmp_dir, f"batch_{batch_num}.csv")
                with open(batch_file, 'w', newline='') as batch_csv:
                    writer = csv.writer(batch_csv)
                    writer.writerow(fieldnames)  # Write header using the provided fieldnames
                    writer.writerows([[row[field] for field in fieldnames] for row in batch])  # Write rows

                batch_num += 1
                print(f"Created batch {batch_num}")

        print(f"Finished preprocessing. Total batches created: {batch_num}")
        return tmp_dir, batch_num

    else:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            header = next(reader)  # Read the header row
            batch_num = 0

            while True:
                batch = list(itertools.islice(reader, batch_size))
                if not batch:
                    break  # Exit the loop if batch is empty

                batch_file = os.path.join(tmp_dir, f"batch_{batch_num}.csv")
                with open(batch_file, 'w', newline='') as batch_csv:
                    writer = csv.writer(batch_csv)
                    writer.writerow(fieldnames)  # Write header using the provided fieldnames
                    writer.writerows([[row[field] for field in fieldnames] for row in batch])  # Write rows

                batch_num += 1
                print(f"Created batch {batch_num}")

    print(f"Finished preprocessing. Total batches created: {batch_num}")
    return tmp_dir, batch_num



def process_batch(batch_file, hexagon_size, feature_colname, x_colname, y_colname, cell_id_colname, quality_colname=None, quality_filter=False, count_colname="NA",smoothing=False, quality_per_hexagon=False, quality_per_probe=False,move_x=0,move_y=0,coord_to_um_conversion=1):
    """
    process_batch(batch_file, hexagon_size, feature_colname, x_colname, y_colname, cell_id_colname, quality_colname=None, quality_filter=False, count_colname="NA", smoothing=False)
    Processes a batch CSV file to calculate hexagon counts and cell counts.

    Args:
    batch_file (str): The path to the batch CSV file.
    hexagon_size (float): The size of the hexagon.
    feature_colname (str): The name of the feature column.
    x_colname (str): The name of the x-coordinate column.
    y_colname (str): The name of the y-coordinate column.
    cell_id_colname (str): The name of the cell ID column.
    quality_colname (str, optional): The name of the quality score column. Defaults to None.
    quality_filter (bool, optional): Whether to filter rows based on quality score. Defaults to False.
    count_colname (str, optional): The name of the count column. Defaults to "NA".
    smoothing (bool, optional): Whether to apply smoothing to the counts. Defaults to False.

    Returns:
        dict or tuple: If cell_id_colname is not "None", returns a tuple containing the hexagon counts and hexagon cell counts.
                    Otherwise, returns the hexagon counts dictionary.

    """
    
    hexagon_counts = {}
    if cell_id_colname != "None":
        hexagon_cell_counts = {}
    if quality_per_hexagon==True:
        hexagon_quality = {}
    if quality_per_probe==True:
        probe_quality = {}
    if count_colname == "NA":
        

        ##########Xenium with quality filter scenario
        with open(batch_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    x = (float(row[x_colname]) + move_x)*coord_to_um_conversion
                    y = (float(row[y_colname]) + move_y)*coord_to_um_conversion
                    closest_hexagon = closest_hex(x, y, hexagon_size)
                    if quality_per_hexagon==True:
                        if closest_hexagon not in hexagon_quality:
                            hexagon_quality[closest_hexagon] = {"mean":float(row[quality_colname]),"count":1}
                        else:
                            hexagon_quality[closest_hexagon]["mean"] = (hexagon_quality[closest_hexagon]["mean"]*hexagon_quality[closest_hexagon]["count"]+float(row[quality_colname]))/(hexagon_quality[closest_hexagon]["count"]+1)
                            hexagon_quality[closest_hexagon]["count"]+=1
                    if quality_per_probe==True:
                        if row[feature_colname] not in probe_quality:
                            probe_quality[row[feature_colname]] = {"mean":float(row[quality_colname]),"count":1}
                        else:
                            probe_quality[row[feature_colname]]["mean"] = (probe_quality[row[feature_colname]]["mean"]*probe_quality[row[feature_colname]]["count"]+float(row[quality_colname]))/(probe_quality[row[feature_colname]]["count"]+1)
                            probe_quality[row[feature_colname]]["count"]+=1

                    if quality_filter==True and float(row[quality_colname]) < 20:
                        continue
                    else:
                        if closest_hexagon not in hexagon_counts:
                            hexagon_counts[closest_hexagon] = {}
                        if row[feature_colname] not in hexagon_counts[closest_hexagon]:
                            hexagon_counts[closest_hexagon][row[feature_colname]] = 0
                        hexagon_counts[closest_hexagon][row[feature_colname]] += 1
                        if cell_id_colname != "None":
                            if closest_hexagon not in hexagon_cell_counts:
                                hexagon_cell_counts[closest_hexagon] = {}
                            if row[cell_id_colname] not in hexagon_cell_counts[closest_hexagon]:
                                hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] = 0
                            hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] += 1
                    

                except ValueError:
                    print(f"Skipping row")
        
    else:
        if smoothing!=False:
            print("Smoothing is on. Counts will be spread into four neighboring squares")
            with open(batch_file, 'r') as file:
                ##########Visium HD with smoothing scenario
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        x = float(row[x_colname])+move_x
                        y = float(row[y_colname])+move_y
                        for x_new,y_new in [(x+smoothing,y+smoothing),(x-smoothing,y-smoothing),(x-smoothing,y+smoothing),(x+smoothing,y-smoothing)]:
                            closest_hexagon = closest_hex(x_new, y_new, hexagon_size)
                            if closest_hexagon not in hexagon_counts:
                                hexagon_counts[closest_hexagon] = {}
                            if row[feature_colname] not in hexagon_counts[closest_hexagon]:
                                hexagon_counts[closest_hexagon][row[feature_colname]] = 0
                            hexagon_counts[closest_hexagon][row[feature_colname]] += float(row[count_colname])/4
                            if cell_id_colname != "None":
                                if closest_hexagon not in hexagon_cell_counts:
                                    hexagon_cell_counts[closest_hexagon] = {}
                                if row[cell_id_colname] not in hexagon_cell_counts[closest_hexagon]:
                                    hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] = 0
                                hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] += float(row[count_colname])/4
                    except ValueError:
                        print(f"Skipping row due to invalid coordinates: {row}")
        else:
            print("No smoothing")
            with open(batch_file, 'r') as file:
                ##########Visium HD scenario without smoothing
                
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        x = float(row[x_colname])+move_x
                        y = float(row[y_colname])+move_y
                        closest_hexagon = closest_hex(x, y, hexagon_size)
                        if closest_hexagon not in hexagon_counts:
                            hexagon_counts[closest_hexagon] = {}
                        if row[feature_colname] not in hexagon_counts[closest_hexagon]:
                            hexagon_counts[closest_hexagon][row[feature_colname]] = 0
                        hexagon_counts[closest_hexagon][row[feature_colname]] += float(row[count_colname])
                        if cell_id_colname != "None":
                            if closest_hexagon not in hexagon_cell_counts:
                                hexagon_cell_counts[closest_hexagon] = {}
                            if row[cell_id_colname] not in hexagon_cell_counts[closest_hexagon]:
                                hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] = 0
                            hexagon_cell_counts[closest_hexagon][row[cell_id_colname]] += float(row[count_colname])
                    except ValueError:
                        print(f"Skipping row due to invalid coordinates: {row}")

    returning_items = [hexagon_counts]

    if cell_id_colname != "None":
        returning_items.append(hexagon_cell_counts)
    if quality_per_hexagon==True:
        returning_items.append(hexagon_quality)
    if quality_per_probe==True:
        returning_items.append(probe_quality)
    return tuple(returning_items)



    


def process_batch_hexagons(batch,hexagon_counts,hexagon_names,features):
        batch_matrix_data = []
        for hexagon in batch:
            count_dict = hexagon_counts.get(hexagon, {})
            hexagon_index = hexagon_names.index(hexagon)
            for feature, count in count_dict.items():
                feature_index = features.index(feature)
                batch_matrix_data.append([feature_index+1, hexagon_index+1, count])
        return batch_matrix_data


def write_10X_h5(adata, file):
    """Writes adata to a 10X-formatted h5 file.
    
    Note that this function is not fully tested and may not work for all cases.
    It will not write the following keys to the h5 file compared to 10X:
    '_all_tag_keys', 'pattern', 'read', 'sequence'

    Args:
        adata (AnnData object): AnnData object to be written.
        file (str): File name to be written to. If no extension is given, '.h5' is appended.

    Raises:
        FileExistsError: If file already exists.

    Returns:
        None
    """
    #remove file
    
    if '.h5' not in file: file = f'{file}.h5'
    if Path(file).exists():
        print(f"File `{file}` already exists. Removing it.")
        os.remove(file)
    def int_max(x):
        return int(max(np.floor(len(str(int(max(x)))) / 4), 1) * 4)
    def str_max(x):
        return max([len(i) for i in x])
    adata.var['feature_types'] = ['Gene Expression' for _ in range(adata.var.shape[0])]
    adata.var['genome'] = ['pv_placeholder' for _ in range(adata.var.shape[0])]
    adata.var['gene_ids'] = adata.var.index

    w = h5py.File(file, 'w')
    grp = w.create_group("matrix")
    grp.create_dataset("barcodes", data=adata.obs_names.values.astype('|S'))

    X = adata.X.T.tocsc()  # Convert the matrix to CSC format
    grp.create_dataset("data", data=X.data.astype(np.float32))
    grp.create_dataset("indices", data=X.indices.astype(np.int32))
    grp.create_dataset("indptr", data=X.indptr.astype(np.int32))
    grp.create_dataset("shape", data=np.array(X.shape).astype(np.int32))

    ftrs = grp.create_group("features")
    if 'feature_types' in adata.var:
        ftrs.create_dataset("feature_type", data=adata.var.feature_types.values.astype('|S'))
    if 'genome' in adata.var:
        ftrs.create_dataset("genome", data=adata.var.genome.values.astype('|S'))
    if 'gene_ids' in adata.var:
        ftrs.create_dataset("id", data=adata.var.gene_ids.values.astype('|S'))
    ftrs.create_dataset("name", data=adata.var.index.values.astype('|S'))
    
    w.close()

    


def process_csv_file(csv_file, hexagon_size, batch_size=1000000, technology="Xenium", feature_colname="feature_name", x_colname="x_location", y_colname="y_location", cell_id_colname="None", quality_colname="qv", max_workers=min(2, multiprocessing.cpu_count()),
                     quality_filter=False, count_colname="NA",smoothing=False,quality_per_hexagon=False,quality_per_probe=False,h5_x_colname="x",h5_y_colname="y",move_x=0,move_y=0,coord_to_um_conversion=1):
    """
    process_csv_file(csv_file, hexagon_size, field_size, batch_size=1000000, technology="Xenium", feature_colname="feature_name", x_colname="x_location", y_colname="y_location", cell_id_colname="None", quality_colname="qv", max_workers=min(2, multiprocessing.cpu_count()), quality_filter=False, count_colname="NA", smoothing=False)
    Processes a CSV file to calculate hexagon counts and cell counts using parallel processing.

    Args:
    csv_file (str): The path to the CSV file.
    hexagon_size (float): The size of the hexagon.
    field_size (tuple): The size of the field as (field_size_x, field_size_y).
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

    Returns:
        tuple: A tuple containing the hexagon counts, hexagons dictionary, and hexagon cell counts.
    """

    print(f"Quality filter is set to {quality_filter}")
    print(f"Quality counting per hexagon is set to {quality_per_hexagon}")
    print(f"Quality counting per probe is set to {quality_per_probe}")

    hexagon_counts = {}
    hexagon_cell_counts = {}
    hexagon_quality = {}
    probe_quality = {}
    #create a nested dict called hexagon quality, with two keys for each hexagon, one for the quality score and one for the count
    
    if technology == "Xenium":
        print("Technology is Xenium. Going forward with default column names.")
        x_colname = "x_location"
        y_colname = "y_location"
        feature_colname = "feature_name"
        cell_id_colname = "cell_id"
        quality_colname = "qv"
        count_colname= "NA"
        coord_to_um_conversion = 1

    elif technology == "Vizgen":
        print("Technology is Vizgen. Going forward with default column names.")
        x_colname = "global_x"
        y_colname = "global_y"
        feature_colname = "gene"
        #cell_id_colname = "barcode_id"
        count_colname= "NA"
        coord_to_um_conversion = 1

    elif technology == "Nanostring":
        print("Technology is Nanostring. Going forward with default column names.")
        x_colname = "x_global_px"
        y_colname = "y_global_px"
        feature_colname = "target"
        cell_id_colname = "cell"
        count_colname= "NA"
        coord_to_um_conversion = 0.12028
        #see ref https://smi-public.objects.liquidweb.services/cosmx-wtx/Pancreas-CosMx-ReadMe.html
        #https://nanostring.com/wp-content/uploads/2023/09/SMI-ReadMe-BETA_humanBrainRelease.html


    elif technology == "Visium_HD":
        print("Technology is Visium_HD. Going forward with pseudovisium processed colnames.")
        x_colname = "x"
        y_colname = "y"
        feature_colname = "gene"
        #cell_id_colname = "barcode"
        count_colname= "count"
        coord_to_um_conversion = 1

    elif technology == "seqFISH":
        print("Technology is seqFISH. Going forward with default column names.")
        x_colname = "x"
        y_colname = "y"
        feature_colname = "name"
        cell_id_colname = "cell"
        count_colname= "NA"
        coord_to_um_conversion = 1
    
    elif technology == "Curio":
        print("Technology is Curio. Going forward with default column names.")
        x_colname = "x"
        y_colname = "y"
        feature_colname = "gene"
        cell_id_colname = "barcode"
        count_colname= "count"
        coord_to_um_conversion = 1


    else:
        print("Technology not recognized. Going forward with set column names.")

    fieldnames=[feature_colname, x_colname, y_colname]
    if cell_id_colname!="None":
        fieldnames.append(cell_id_colname)
    if quality_filter==True or quality_per_hexagon==True or quality_per_probe==True:
        fieldnames.append(quality_colname)
    if count_colname!="NA":
        fieldnames.append(count_colname)



    tmp_dir, num_batches = preprocess_csv(csv_file, batch_size, fieldnames)
    batch_files = [os.path.join(tmp_dir, f"batch_{i}.csv") for i in range(num_batches)]
    n_process = min(max_workers, multiprocessing.cpu_count())
    print(f"Processing batches using {n_process} processes")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = [executor.submit(process_batch, batch_file, hexagon_size, feature_colname, x_colname, y_colname, cell_id_colname, quality_colname, quality_filter, count_colname,smoothing,quality_per_hexagon,quality_per_probe, move_x,move_y,coord_to_um_conversion) for batch_file in batch_files]
        
        with tqdm(total=len(batch_files), desc="Processing batches", unit="batch") as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                all_res = future.result()
                batch_hexagon_counts = all_res[0]
                for hexagon_counts_hex, counts in batch_hexagon_counts.items():
                    for feature_name, count in counts.items():
                        try:
                            if hexagon_counts_hex not in hexagon_counts:
                                hexagon_counts[hexagon_counts_hex] = {}
                            if feature_name not in hexagon_counts[hexagon_counts_hex]:
                                hexagon_counts[hexagon_counts_hex][feature_name] = 0
                            hexagon_counts[hexagon_counts_hex][feature_name] += count
                            
                                
                        except KeyError:
                            print(f"Error in trying to add to hexagon_counts")

                if cell_id_colname != "None":
                    batch_hexagon_cell_counts = all_res[1]
                    for hexagon_cell_counts_hex, cell_counts in batch_hexagon_cell_counts.items():
                        for cell_id, cell_count in cell_counts.items():
                            try:
                                if hexagon_cell_counts_hex not in hexagon_cell_counts:
                                    hexagon_cell_counts[hexagon_cell_counts_hex] = {}
                                if cell_id not in hexagon_cell_counts[hexagon_cell_counts_hex]:
                                    hexagon_cell_counts[hexagon_cell_counts_hex][cell_id] = 0
                                hexagon_cell_counts[hexagon_cell_counts_hex][cell_id] += cell_count
                            except KeyError:
                                print(f"Error in trying to add to hexagon_cell_counts")

                if quality_per_hexagon==True:
                    if cell_id_colname != "None":
                        batch_hexagon_quality = all_res[2]
                    else:
                        batch_hexagon_quality = all_res[1]
                    for hexagon_quality_hex, quality_dict in batch_hexagon_quality.items():
                        try:
                            if hexagon_quality_hex not in hexagon_quality:
                                hexagon_quality[hexagon_quality_hex] = quality_dict
                            else:
                                hexagon_quality[hexagon_quality_hex]["mean"] = (float(hexagon_quality[hexagon_quality_hex]["mean"]) * hexagon_quality[hexagon_quality_hex]["count"] + float(quality_dict["mean"]) * quality_dict["count"]) / (hexagon_quality[hexagon_quality_hex]["count"] + quality_dict["count"])
                                hexagon_quality[hexagon_quality_hex]["count"] += quality_dict["count"]
                        except KeyError:
                            print(f"Error in trying to add to hexagon_quality")
                
                if quality_per_probe==True:
                    if cell_id_colname != "None" and quality_per_hexagon==True:
                        batch_probe_quality = all_res[3]
                    elif cell_id_colname != "None" and quality_per_hexagon==False:
                        batch_probe_quality = all_res[2]
                    elif cell_id_colname == "None" and quality_per_hexagon==True:
                        batch_probe_quality = all_res[2]
                    else:
                        batch_probe_quality = all_res[1]
                

                    for probe, quality_dict in batch_probe_quality.items():
                        try:
                            if probe not in probe_quality:
                                probe_quality[probe] = quality_dict
                            else:
                                probe_quality[probe]["mean"] = (float(probe_quality[probe]["mean"]) * probe_quality[probe]["count"] + float(quality_dict["mean"]) * quality_dict["count"]) / (probe_quality[probe]["count"] + quality_dict["count"])
                                probe_quality[probe]["count"] += quality_dict["count"]
                        except KeyError:
                            print(f"Error in trying to add to probe_quality")
                progress_bar.update(1)

    shutil.rmtree(tmp_dir)  # Remove temporary directory and files

    return hexagon_counts, hexagon_cell_counts, hexagon_quality, probe_quality



def process_hexagon(hexagon, hexagon_index, features, hexagon_counts):
        return [[features.index(feature) + 1, hexagon_index + 1, count]
                for feature, count in hexagon_counts[hexagon].items()]

def hex_to_rows(hexagon_batch, start_index, features, hexagon_counts):
    matrix_data = []
    for hexagon_index, hexagon in enumerate(hexagon_batch, start=start_index):
        count_dict = hexagon_counts.get(hexagon, {})
        for feature, count in count_dict.items():
            feature_index = features.index(feature)
            matrix_data.append([feature_index + 1, hexagon_index + 1, count])
    print(f"Batch total count: {sum(row[2] for row in matrix_data)}")
    return matrix_data

def create_pseudovisium(path,hexagon_counts,hexagon_cell_counts,hexagon_quality, probe_quality,
                        img_file_path=None,  project_name="project",
                         alignment_matrix_file=None,image_pixels_per_um=1/0.2125,hexagon_size=100,tissue_hires_scalef=0.2,
                         pixel_to_micron=False,max_workers=min(2, multiprocessing.cpu_count())):
    """
    create_pseudovisium(path, hexagon_counts, hexagon_cell_counts, img_file_path=None, project_name="project", alignment_matrix_file=None, image_pixels_per_um=1/0.2125, hexagon_size=100, tissue_hires_scalef=0.2, pixel_to_micron=False, max_workers=min(2, multiprocessing.cpu_count()))
    Creates a Pseudovisium output directory structure and files.
    
    Args:
    path (str): The path to create the Pseudovisium output directory.
    hexagon_counts (dict): A dictionary of hexagon counts.
    hexagon_cell_counts (dict): A dictionary of hexagon cell counts.
    img_file_path (str, optional): The path to the image file. Defaults to None.
    project_name (str, optional): The name of the project. Defaults to "project".
    alignment_matrix_file (str, optional): The path to the alignment matrix file. Defaults to None.
    image_pixels_per_um (float, optional): The number of image pixels per micrometer. Defaults to 1/0.2125.
    hexagon_size (int, optional): The size of the hexagon. Defaults to 100.
    tissue_hires_scalef (float, optional): The scaling factor for the high-resolution tissue image. Defaults to 0.2.
    pixel_to_micron (bool, optional): Whether to convert pixel coordinates to micron coordinates. Defaults to False.
    max_workers (int, optional): The maximum number of worker processes to use. Defaults to min(2, multiprocessing.cpu_count()).
    """

    #to path, create a folder called pseudovisium
    folderpath = path+ '/pseudovisium/' + project_name
    #if folderpath exists, delete it
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
    try:
        print("Creating pseudovisium folder in output path:{0}".format(folderpath))
 
        if os.path.exists(path+ '/pseudovisium/'):
            print("Using already existing folder: {0}".format(path+ '/pseudovisium/'))
        else:
            os.mkdir(path+ '/pseudovisium/')
        os.mkdir(folderpath)
        os.mkdir(folderpath + '/spatial')
    except:
        pass
    
 ############################################## ##############################################
    # see https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors
    #https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
    scalefactors = {"tissue_hires_scalef":tissue_hires_scalef,
                     "tissue_lowres_scalef": tissue_hires_scalef/10,
                       "fiducial_diameter_fullres": 0,
                         "spot_diameter_fullres": 2*hexagon_size*(image_pixels_per_um)}
    
    print("Creating scalefactors_json.json file in spatial folder.")
    with open(folderpath +'/spatial/scalefactors_json.json', 'w') as f:
        json.dump(scalefactors, f)
 ############################################## ##############################################

    x, y, x_, y_, contain, hexagon_names = [], [], [], [], [], []
    for hexagon in hexagon_counts:
        x_.append((hexagon[0] + hexagon_size) // (2 * hexagon_size))
        y_.append(hexagon[1] // (1.73205 * hexagon_size))
        x.append(hexagon[0])
        y.append(hexagon[1])
        contain.append(1 if sum(hexagon_counts[hexagon].values()) > hexagon_size else 0)
        hexagon_names.append(hexagon)
    
 ############################################## ##############################################
    barcodes = ["hexagon_{}".format(i) for i in range(1, len(hexagon_names) + 1)]
    barcodes_table = pd.DataFrame({'barcode':barcodes})
    #save to pseudo visium root
    barcodes_table.to_csv(folderpath+'/barcodes.tsv',sep='\t',index=False,header=False)

    print("Creating barcodes.tsv.gz file in spatial folder.")
    with open(folderpath +'/barcodes.tsv', 'rb') as f_in:
        with gzip.open(folderpath +'/barcodes.tsv.gz', 'wb') as f_out:
            f_out.writelines(f_in)
 ############################################## ##############################################
    hexagon_table = pd.DataFrame(zip(barcodes, contain, y_, x_, 
                                 [int(image_pixels_per_um * a) for a in y],
                                 [int(image_pixels_per_um * a) for a in x]),
                             columns=['barcode', 'in_tissue', 'array_row', 'array_col',
                                      'pxl_row_in_fullres', 'pxl_col_in_fullres'])
    
    print("Creating tissue_positions_list.csv file in spatial folder.")
    hexagon_table.to_csv(folderpath +'/spatial/tissue_positions_list.csv',index=False,header=False)

 ############################################## ##############################################
    #if hexagon_cell_counts is empty, then skip
    if hexagon_cell_counts == {}:
        print("No cell information provided. Skipping cell information files.")
    else:
        print("Creating pv_cell_hex.csv file in spatial folder.")
        with open(folderpath + '/spatial/pv_cell_hex.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for hexagon, cell_count_dict in hexagon_cell_counts.items():
                hexagon_index = hexagon_names.index(hexagon)
                for cell, count in cell_count_dict.items():
                    writer.writerow([cell, hexagon_index + 1, count])

 ############################################## ##############################################
    if hexagon_quality == {}:
        print("No quality information provided. Skipping quality information files.")
    else:
        print("Creating quality_per_hexagon.csv file in spatial folder.")
        with open(folderpath + '/spatial/quality_per_hexagon.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for hexagon, quality_dict in hexagon_quality.items():
                try:
                    hexagon_index = hexagon_names.index(hexagon)
                    writer.writerow([hexagon_index + 1, quality_dict["mean"], quality_dict["count"]])
                except ValueError:
                    print(f"""One hexagon quality measurement skipped, with mean {quality_dict['mean']} and count {quality_dict['count']}, as no
                          actual counts were found for this hexagon.""")

 ############################################## ##############################################
    if probe_quality == {}:
        print("No quality information provided. Skipping quality information files.")
    else:
        print("Creating quality_per_probe.csv file in spatial folder.")
        with open(folderpath + '/spatial/quality_per_probe.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for probe, quality_dict in probe_quality.items():
                writer.writerow([probe, quality_dict["mean"], quality_dict["count"]])

 ############################################## ##############################################

    features = list(set(feature for hexagon_counts in hexagon_counts.values() for feature in hexagon_counts)) 
    # Create a list of rows with repeated features and 'Gene Expression' column
    rows = [[feature, feature, 'Gene Expression'] for feature in features]

    print("Creating features.tsv.gz file in spatial folder.")
    with open(folderpath + '/features.tsv', 'wt', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerows(rows)
    
    # Create a features.tsv.gz file
    with open(folderpath + '/features.tsv', 'rb') as f_in, gzip.open(folderpath + '/features.tsv.gz', 'wb') as f_out:
        f_out.writelines(f_in)
        

 ############################################## ##############################################
    

    # Assuming hexagon_counts is your dictionary
    ordered_hexagon_counts = dict(sorted(hexagon_counts.items()))

    print("Putting together the matrix.mtx file")
    matrix_data = []

    n_total_hexagons = len(ordered_hexagon_counts)

    total_count = 0
    n_processes = min(max_workers, multiprocessing.cpu_count())
    batch_size_n_hexagons = n_total_hexagons // (n_processes * 4)
    print(f"Using {n_processes} processes")
    print(f"Processing {n_total_hexagons} hexagons in batches of {batch_size_n_hexagons} hexagons")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = []
        hexagon_names = list(ordered_hexagon_counts.keys())
        for i in range(0, n_total_hexagons, batch_size_n_hexagons):
            hexagon_batch = hexagon_names[i:i + batch_size_n_hexagons]
            future = executor.submit(hex_to_rows, hexagon_batch, i, features, ordered_hexagon_counts)
            futures.append(future)

        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future in concurrent.futures.as_completed(futures):
                batch_matrix_data = future.result()
                matrix_data.extend(batch_matrix_data)
                total_count += sum(row[2] for row in batch_matrix_data)
                pbar.update(1)

    print(f"Total matrix count: {total_count}")
    unique_hexagons = len(set(hexagon_names))
    print(f"Number of unique hexagons: {unique_hexagons}")
    print(f"Number of hexagons in ordered_hexagon_counts: {len(ordered_hexagon_counts)}")

    

    data = np.array(matrix_data)[:, 2]
    row_indices = np.array(matrix_data)[:, 0] - 1
    col_indices = np.array(matrix_data)[:, 1] - 1

    sparse_matrix = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(len(features), len(barcodes)))

    print("Creating matrix.mtx.gz file in spatial folder.")
    with open(folderpath + '/matrix.mtx', 'wb') as f:
        scipy.io.mmwrite(f, sparse_matrix, comment='metadata_json: {"software_version": "Pseudovisium", "format_version": 1}\n')

    with open(folderpath +'/matrix.mtx', 'rb') as f_in, gzip.open(folderpath + '/matrix.mtx.gz', 'wb') as f_out:
        f_out.writelines(f_in)
    
 ############################################## ##############################################



    print("Putting together the filtered_feature_bc_matrix.h5 file")
    data = np.array(matrix_data)[:, 2]
    row_indices = np.array(matrix_data)[:, 1] - 1  # Use hexagon indices for rows
    col_indices = np.array(matrix_data)[:, 0] - 1  # Use feature indices for columns
    sparse_matrix = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(len(barcodes), len(features)))

    # Create AnnData object from sparse matrix and barcodes/features
    adata = sc.AnnData(X=sparse_matrix, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=features))

    # Write AnnData object to 10X-formatted h5 file
    write_10X_h5(adata, folderpath + '/filtered_feature_bc_matrix.h5')

############################################## ##############################################

    #check if alignment matrix file is given
    if alignment_matrix_file:
        print("Alignment matrix found and will be used to create tissue_hires_image.png and tissue_lowres_image.png files in spatial folder.")
        M = pd.read_csv(alignment_matrix_file,header=None,index_col=None).to_numpy()
    else:
        M = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
            # Load the H&E image
    if img_file_path:
        #if img_filepath is tiff
        if img_file_path.endswith('.tiff') or img_file_path.endswith('.tif'):
            image = tifffile.imread(img_file_path)
        #elif png
        elif img_file_path.endswith('.png'):
            image = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)
        #resizing the image according to tissue_hires_scalef, but such that it also satisfies the incoming
        #scaling factor from the alignment matrix

        if pixel_to_micron:
            M[0, 0] = 1/M[0, 0]   # Update x-scale to 1 because rescaling already done here
            M[1, 1] = 1/M[1, 1]   # Update y-scale to 1 because rescaling already done here
            width = int(image.shape[1] * tissue_hires_scalef) #* M[0, 0])  # New width
            height = int(image.shape[0] * tissue_hires_scalef) #* M[0, 0])  # New height
            dim = (width, height)
        # resize image
            resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            #update translation
            M[0, 2] *= tissue_hires_scalef * M[0, 1]
            M[1, 2] *= tissue_hires_scalef * M[0, 1]

            M[0, 2] = -M[0, 2]
            M[1, 2] = -M[1, 2]

        else: #normal scenario

            width = int(image.shape[1] * tissue_hires_scalef )  # New width
            height = int(image.shape[0] * tissue_hires_scalef)  # New height
            dim = (width, height)
            # resize image
            resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            #update translation
            M[0, 2] *= tissue_hires_scalef #* max(M[0, 0], M[1, 1])  # Update x-translation
            M[1, 2] *= tissue_hires_scalef #* max(M[0, 0], M[1, 1])  # Update y-translation

        max_dim = max(resized_img.shape)
        max_stretch = max(M[0, 0], M[1, 1],M[1, 0],M[0, 1] )*1.2
                # Apply the transformation
        new_width = int(max_dim*max_stretch)
        new_height = int(max_dim*max_stretch)
        image = cv2.warpAffine(resized_img, M[:2], (new_width, new_height))


        #and to 2%
        scale2 = 0.1
        width = int(image.shape[1] * scale2)
        height = int(image.shape[0] * scale2)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        #finally if image is a single channel, convert to three channels, because some packages will expect that
        #when importing.
        dims = len(image.shape)
        #if 2, then triple the image to make it thre channels
        if dims == 2:
            image = np.array([image,image,image])
            #change order of axis
            image = np.moveaxis(image,0,-1)

            resized = np.array([resized,resized,resized])
            #change order of axis
            resized = np.moveaxis(resized,0,-1)


        print("Creating tissue_hires_image.png file in spatial folder.")
        #save as 8bit
        cv2.imwrite(folderpath +'/spatial/tissue_hires_image.png', image/np.max(image)*255)
        cv2.imwrite(folderpath +'/spatial/tissue_lowres_image.png', resized/np.max(resized)*255)
    
    else:
        #output a blank image
        print("No image file provided. Creating blank tissue_hires_image.png and tissue_lowres_image.png files in spatial folder.")
        image = np.zeros((1000,1000))
        cv2.imwrite(folderpath +'/spatial/tissue_hires_image.png', image)
        cv2.imwrite(folderpath +'/spatial/tissue_lowres_image.png', image)
        


#####################
###############  Visium HD/Slide-seq helper functions 
###################################################################################################################################################
def read_files(folder,technology):
    """
    read_files(folder)
    Reads the necessary files from a Visium HD folder.

    Args:
    folder (str): The path to the Visium HD folder.

    Returns:
        tuple: A tuple containing the scale factors, tissue positions, and filtered feature-barcode matrix.
    """
    if technology == "Visium_HD":
        scalefactors = json.load(open(folder+"/spatial/scalefactors_json.json"))
        tissue_pos = pd.read_parquet(folder+"/spatial/tissue_positions.parquet")
        fb_matrix = sc.read_10x_h5(folder+"/filtered_feature_bc_matrix.h5")
        return scalefactors,tissue_pos,fb_matrix
    elif technology == "Curio":
        #find file ending with .h5ad in folder
        folder_files = os.listdir(folder)
        h5ad_file = [file for file in folder_files if file.endswith(".h5ad")][0]
        adata = sc.read(folder+h5ad_file)
        return adata
    

def anndata_to_df(adata, technology,tissue_pos=None,scalefactors=None,x_col=None,y_col=None):
    """
    anndata_to_df(adata, tissue_pos, scalefactors)
    Converts an AnnData object to a DataFrame.
    Args:
    adata (AnnData): The AnnData object containing the counts matrix.
    tissue_pos (DataFrame): The tissue positions DataFrame.
    scalefactors (dict): The scale factors dictionary.

    Returns:
        tuple: A tuple containing the converted DataFrame and the image resolution.
    """
    if technology == "Visium_HD":
        #get image resolution of the hires image
        image_resolution = scalefactors["microns_per_pixel"]/scalefactors["tissue_hires_scalef"]
        #change to pixel per micron
        image_resolution = 1/image_resolution
        #tissue_pos keep barcode, pxl_row_in_fullres, pxl_col_in_fullres
        tissue_pos = tissue_pos[["barcode","pxl_row_in_fullres","pxl_col_in_fullres"]]
        # Convert the AnnData matrix to a sparse matrix (CSR format)
        X = scipy.sparse.csr_matrix(adata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()

        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame({
            'barcode': adata.obs_names[row_indices],
            'gene': adata.var_names[col_indices],
            'count': X.data
        })

        #join with tissue_pos
        df = pd.merge(df,tissue_pos,on="barcode")

        #extract microns_per_pixel
        microns_per_pixel = scalefactors["microns_per_pixel"]
        #convert pxl_row_in_fullres and pxl_col_in_fullres to microns
        df["y"] = df["pxl_row_in_fullres"]*microns_per_pixel
        df["x"] = df["pxl_col_in_fullres"]*microns_per_pixel
        #remove pxl_row_in_fullres and pxl_col_in_fullres
        df.drop(["pxl_row_in_fullres","pxl_col_in_fullres"],axis=1,inplace=True)

        # Write the DataFrame to a CSV file
        return df,image_resolution 
    elif technology == "Curio":
        #get scales
        
        #keep only the columns x,y
        obs = adata.obs[[x_col,y_col]]
        #rename these cols to x and y
        obs.columns = ["x","y"]
        scale = np.round(min([abs(x1 - x2) for x1 in obs["y"] for x2 in [obs["y"][500]] if x1!=x2]),3) #should be 10um
        
        obs["barcode"] = obs.index
        X = scipy.sparse.csr_matrix(adata.X)

        # Get the row and column indices of non-zero elements
        row_indices, col_indices = X.nonzero()
        genes = adata.var.index if (adata.var.index[0] != 0) and (adata.var.index[0] != "0") else adata.var[adata.var.columns[0]]
        genes = genes[col_indices]
        # Create a DataFrame with the row names, column names, and counts
        df = pd.DataFrame({
            'barcode': adata.obs_names[row_indices],
            'gene': genes,
            'count': X.data
        })
        #join with tissue_pos
        df = pd.merge(df,obs,on="barcode")
        return df,scale


def visium_hd_curio_to_transcripts(folder,output,technology,x_col=None,y_col=None):
    """
    visium_hd_to_transcripts(folder, output)
    Converts Visium HD files to a transcripts CSV file.
    Args:
    folder (str): The path to the Visium HD folder.
    output (str): The path to save the transcripts CSV file.

    Returns:
        float: The image resolution (pixels per micrometer).
    """

    if technology == "Visium_HD":
        scalefactors,tissue_pos,fb_matrix = read_files(folder,technology)
        df,image_resolution  = anndata_to_df(adata=fb_matrix,technology=technology,tissue_pos=tissue_pos,scalefactors=scalefactors)
        df.to_csv(output,index=False)
        return image_resolution 
    elif technology == "Curio":
        adata = read_files(folder,technology)
        df,scale = anndata_to_df(adata=adata,technology=technology,x_col=x_col,y_col=y_col)
        df.to_csv(output,index=False)
        return scale





######### Main function to generate pseudovisium output ############################################################################################################
    
def generate_pv(csv_file,img_file_path=None, hexagon_size=100,  output_path=None, batch_size=1000000, alignment_matrix_file=None, project_name='project',
                image_pixels_per_um=1/0.85, tissue_hires_scalef=0.2,technology="Xenium", 
                feature_colname="feature_name", x_colname="x_location", y_colname="y_location",
                cell_id_colname="None", quality_colname="qv",
                pixel_to_micron=False, max_workers=min(2, multiprocessing.cpu_count()), quality_filter=False, count_colname="NA",visium_hd_folder=None,
                smoothing=False,quality_per_hexagon=False,quality_per_probe=False,h5_x_colname = "x", h5_y_colname = "y",move_x=0,move_y=0,coord_to_um_conversion=1):
    """
    generate_pv(csv_file, img_file_path=None, hexagon_size=100, field_size_x=1000, field_size_y=1000, output_path=None, 
    batch_size=1000000, alignment_matrix_file=None, project_name='project', image_pixels_per_um=1/0.85, tissue_hires_scalef=0.2, 
    technology="Xenium", feature_colname="feature_name", x_colname="x_location", y_colname="y_location", cell_id_colname="None", 
    quality_colname="qv", pixel_to_micron=False, max_workers=min(2, multiprocessing.cpu_count()), quality_filter=False, 
    count_colname="NA", visium_hd_folder=None, smoothing=False)

    
    Generates a Pseudovisium output from a CSV file.

    Args:
    csv_file (str): The path to the CSV file.
    img_file_path (str, optional): The path to the image file. Defaults to None.
    hexagon_size (int, optional): The size of the hexagon. Defaults to 100.
    field_size_x (int, optional): The size of the field in the x-dimension. Defaults to 1000.
    field_size_y (int, optional): The size of the field in the y-dimension. Defaults to 1000.
    output_path (str, optional): The path to save the Pseudovisium output. Defaults to None.
    batch_size (int, optional): The number of rows per batch. Defaults to 1000000.
    alignment_matrix_file (str, optional): The path to the alignment matrix file. Defaults to None.
    project_name (str, optional): The name of the project. Defaults to 'project'.
    image_pixels_per_um (float, optional): The number of image pixels per micrometer. Defaults to 1/0.85.
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
    """


    start = time.time()

    if technology == "Visium_HD":
        print("Technology is Visium_HD. Generating transcripts.csv file from Visium HD files.")
        #Automatically calculating image_pixels_per_um from the scalefactors_json.json file
        image_pixels_per_um = visium_hd_curio_to_transcripts(visium_hd_folder,visium_hd_folder+"/transcripts.csv",technology)
        csv_file = visium_hd_folder+"/transcripts.csv"
    if technology == "Curio":
        print("Technology is Curio. Generating transcripts.csv file from Curio files.")
        smoothing_scale = visium_hd_curio_to_transcripts(visium_hd_folder,visium_hd_folder+"/transcripts.csv",technology,x_col=h5_x_colname,y_col=h5_y_colname)
        csv_file = visium_hd_folder+"/transcripts.csv"
        print("Smoothing defaults to : {0}".format(smoothing_scale/4))
        smoothing = smoothing_scale/4
        
        # Process CSV file to generate hexagon counts and hexagon information
    
    hexagon_counts, hexagon_cell_counts, hexagon_quality, probe_quality= process_csv_file(csv_file, hexagon_size,  batch_size, 
             technology, feature_colname, x_colname, y_colname,cell_id_colname, quality_colname=quality_colname,
               max_workers=max_workers, quality_filter=quality_filter, count_colname=count_colname,smoothing=smoothing,
               quality_per_hexagon=quality_per_hexagon,quality_per_probe=quality_per_probe,h5_x_colname=h5_x_colname,h5_y_colname=h5_y_colname,move_x=move_x,move_y=move_y,coord_to_um_conversion=coord_to_um_conversion)
        
    # Create Pseudovisium output
    create_pseudovisium(path=output_path,hexagon_counts=hexagon_counts, hexagon_cell_counts=hexagon_cell_counts, probe_quality=probe_quality,
                        img_file_path=img_file_path, hexagon_quality =hexagon_quality,
                          project_name=project_name, alignment_matrix_file=alignment_matrix_file,
                          image_pixels_per_um=image_pixels_per_um,hexagon_size=hexagon_size,
                          tissue_hires_scalef=tissue_hires_scalef,pixel_to_micron=pixel_to_micron,max_workers=max_workers)

    #save all arguments in a json file called arguments.json
    print("Creating arguments.json file in output path.")
    arguments = {"csv_file":csv_file,"img_file_path":img_file_path,"hexagon_size":hexagon_size,
                    "output_path":output_path,
                    "batch_size":batch_size,"alignment_matrix_file":alignment_matrix_file,"project_name":project_name,
                    "image_pixels_per_um":image_pixels_per_um,"tissue_hires_scalef":tissue_hires_scalef,
                    "technology":technology,"feature_colname":feature_colname,"x_colname":x_colname,
                    "y_colname":y_colname,"cell_id_colname":cell_id_colname,"pixel_to_micron":pixel_to_micron,
                    "quality_colname":quality_colname,"quality_filter":quality_filter,"count_colname":count_colname,
                    "smoothing":smoothing,"quality_per_hexagon":quality_per_hexagon,"quality_per_probe":quality_per_probe,
                    "max_workers":max_workers,"visium_hd_folder":visium_hd_folder,"h5_x_colname":h5_x_colname,"h5_y_colname":h5_y_colname,
                    "move_x":move_x,"move_y":move_y,"coord_to_um_conversion":coord_to_um_conversion}

    with open(output_path + '/pseudovisium/' + project_name + '/arguments.json', 'w') as f:
        json.dump(arguments, f)

    end = time.time()
    print(f"Time taken: {end - start} seconds")




def main():
    """
    main()
    The main function that parses command-line arguments and calls the generate_pv function.
    """
    parser = argparse.ArgumentParser(description="Process parameters.")
    parser.add_argument("--csv_file", "-c", type=str, help="CSV file path", default=None)
    parser.add_argument("--output_path", "-o", type=str, help="Output path", default='.')
    parser.add_argument("--hexagon_size", "-hs", type=int, help="Hexagon size", default=100)
    parser.add_argument("--img_file_path", "-i", type=str, help="Image file path", default=None)
    parser.add_argument("--alignment_matrix_file", "-am", type=str, help="Alignment matrix file path", default=None)
    parser.add_argument("--batch_size", "-b", type=int, help="Batch size", default=1000000)
    parser.add_argument("--project_name", "-p", type=str, help="Project name", default='project')
    parser.add_argument("--image_pixels_per_um", "-ppu", type=float, help="Image pixels per um", default=1/0.2125)#change!
    parser.add_argument("--tissue_hires_scalef", "-ths", type=float, help="Tissue hires scale factor", default=0.2)
    parser.add_argument("--technology", "-t", type=str, help="Technology", default="Xenium")
    parser.add_argument("--feature_colname", "-fc", type=str, help="Feature column name", default="feature_name")
    parser.add_argument("--x_colname", "-xc", type=str, help="X column name", default="x_location")
    parser.add_argument("--y_colname", "-yc", type=str, help="Y column name", default="y_location")
    parser.add_argument("--cell_id_colname", "-cc", type=str, help="Cell ID column name", default="None")
    parser.add_argument("--pixel_to_micron", "-ptm", action="store_true", help="Convert pixel to micron")
    parser.add_argument("--quality_colname", "-qcol", type=str, help="Quality column name", default="qv")
    parser.add_argument("--count_colname", "-ccol", type=str, help="Count column name", default="NA")
    parser.add_argument("--smoothing", "-s", type=float, help="Smoothing factor", default=0.0)
    parser.add_argument("--visium_hd_folder", "-vhf", type=str, help="Visium HD folder", default=None)
    parser.add_argument("--mw", "--max_workers", type=int, help="Max workers", default=min(2, multiprocessing.cpu_count()))
    parser.add_argument("--quality_filter", "-qf", action="store_true", help="Filter out rows with quality score less than 20")
    parser.add_argument("--quality_per_hexagon", "-qph", action="store_true", help="Calculate quality per hexagon")
    parser.add_argument("--quality_per_probe", "-qpp", action="store_true", help="Calculate quality per probe")
    parser.add_argument("--h5_x_colname", "-h5x", type=str, help="X column name in h5ad file", default="x")
    parser.add_argument("--h5_y_colname", "-h5y", type=str, help="Y column name in h5ad file", default="y")
    parser.add_argument("--move_x","-mx", type=int, help="Move x", default=0)
    parser.add_argument("--move_y","-my", type=int, help="Move y", default=0)
    parser.add_argument("--coord_to_um_conversion","-ctu", type=float, help="Conversion factor from coordinates to microns", default=1.0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print out script purpose and parameters")
    
    
    #make sure to add verbose as well
    parser.add_argument("-help", action="store_true", help="Print out script purpose and parameters")
    args = parser.parse_args()

    if args.help:
        print("This is Pseudovisium, a software that compresses imaging-based spatial transcriptomics files using hexagonal binning of the data.")
        parser.print_help()
        sys.exit(0)
    

    generate_pv(csv_file=args.csv_file,img_file_path=args.img_file_path,
                hexagon_size=args.hexagon_size, output_path=args.output_path, 
                batch_size=args.batch_size,
                alignment_matrix_file=args.alignment_matrix_file, 
                project_name=args.project_name,image_pixels_per_um=args.image_pixels_per_um, 
                tissue_hires_scalef=args.tissue_hires_scalef,technology=args.technology, 
                feature_colname=args.feature_colname, x_colname=args.x_colname, 
                y_colname=args.y_colname,cell_id_colname=args.cell_id_colname,
                pixel_to_micron=args.pixel_to_micron,max_workers=args.mw,
                quality_colname=args.quality_colname,quality_filter=args.quality_filter,
                count_colname=args.count_colname,
                smoothing=args.smoothing,
                quality_per_hexagon=args.quality_per_hexagon,
                quality_per_probe=args.quality_per_probe,
                h5_x_colname=args.h5_x_colname,h5_y_colname=args.h5_y_colname,
                move_x=args.move_x,move_y=args.move_y,
                coord_to_um_conversion=args.coord_to_um_conversion,
                visium_hd_folder=args.visium_hd_folder)
                
    print("Pseudovisium output generated successfully.")


    


if __name__ == "__main__":
    main()

