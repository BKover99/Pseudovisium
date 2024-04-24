import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import os
import shutil
import gzip
import cv2
import scanpy as sc
import scipy.io

def from_h5_to_files(fold):
    ad = sc.read_10x_h5(fold + "/filtered_feature_bc_matrix.h5")
    features = pd.DataFrame(ad.var.index)
    features[1] = features[0]
    features[2] = "Gene Expression"
    features.to_csv(os.path.join(fold, "features.tsv"), sep="\t", index=False, header=False)
    pd.DataFrame(ad.obs.index).to_csv(os.path.join(fold, "barcodes.tsv"), sep="\t", index=False, header=False)
    scipy.io.mmwrite(os.path.join(fold, "matrix.mtx"), ad.X.T)

def load_data(folder):
    dataset_name = folder.split("/")[-2] 
    print(f"Loading in {dataset_name}")
    
    barcodes_file = folder + "/barcodes.tsv"
    features_file = folder + "/features.tsv"
    matrix_file = folder + "/matrix.mtx"
    
    if not os.path.exists(barcodes_file) or not os.path.exists(features_file) or not os.path.exists(matrix_file):
        print(f"Missing files in {folder}. Attempting to generate from filtered_feature_bc_matrix.h5.")
        from_h5_to_files(folder)
    
    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")
    barcodes.columns = ["Barcode_ID"]
    barcodes["Barcode_ID"] = [f"{barcode}_{dataset_name}" for barcode in barcodes["Barcode_ID"]]
    barcodes["index_col"] = barcodes.index + 1
    
    tissue_positions_list = pd.read_csv(folder + "/spatial/tissue_positions_list.csv", header=None)
    tissue_positions_list.columns = ["barcode", "in_tissue", "tissue_col", "tissue_row", "x", "y"]
    tissue_positions_list["barcode"] = [f"{barcode}_{dataset_name}" for barcode in tissue_positions_list["barcode"]]
    tissue_positions_list["dataset"] = dataset_name
    barcode_indices = []
    for barcode in tissue_positions_list["barcode"]:
        try:
            barcode_indices.append(barcodes[barcodes["Barcode_ID"] == barcode].index_col.values[0])
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
    
    image_exists = os.path.exists(folder + "/spatial/tissue_hires_image.png")
    if image_exists:
        image = plt.imread(folder + "/spatial/tissue_hires_image.png")
    else:
        image = np.zeros((int(tissue_positions_list["x"].max()), int(tissue_positions_list["y"].max())))
    
    scalefactors = json.load(open(folder + "/spatial/scalefactors_json.json"))
    
    data = {
        "matrix": matrix,
        "tissue_positions_list": tissue_positions_list, 
        "features": features,
        "barcodes": barcodes,
        "image": image,
        "scalefactors": scalefactors
    }
    return dataset_name, data



def merge_data(folders, pv_format=False):
    dataset_names = []
    nested_dict = {}
    features_df = pd.DataFrame()
    barcodes_df = pd.DataFrame()
    
    scalefactor_hires_vals=[]
    for folder in folders:
        scalefactors = json.load(open(folder+"/spatial/scalefactors_json.json"))
        scalefactor_hires_vals.append(scalefactors["tissue_hires_scalef"])
    
    min_scalefactor = min(scalefactor_hires_vals)
    resize_factors=[]
    for scalefactor_hires in scalefactor_hires_vals:
        resize_factors.append(min_scalefactor/scalefactor_hires)
    
    scalefactor_hires = min_scalefactor 
    min_scalefactor_folder = folders[scalefactor_hires_vals.index(min_scalefactor)]
    scalefactors_for_save = json.load(open(min_scalefactor_folder+"/spatial/scalefactors_json.json"))
    
    max_x_range = 0
    max_y_range = 0
    max_col_dims = 0
    max_row_dims = 0
    
    for folder,i in zip(folders,range(len(folders))):
        dataset_name, data = load_data(folder)
        dataset_names.append(dataset_name)
        nested_dict[dataset_name] = data

        #adjust image size
        image = data["image"]
        width = int(image.shape[1] * resize_factors[i])
        height = int(image.shape[0] * resize_factors[i])
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        nested_dict[dataset_name]["image"] = image
        
        barcodes_df_to_add = data["barcodes"].copy()
        barcodes_df_to_add["dataset"] = dataset_name
        barcodes_df = pd.concat([barcodes_df,barcodes_df_to_add])
        
        features_df_to_add = data["features"].copy()
        features_df_to_add["dataset"] = dataset_name
        features_df = pd.concat([features_df,features_df_to_add])
        
        if pv_format:
            pv_cell_hex = pd.read_csv(folder+"/spatial/pv_cell_hex.csv",header=None)
            pv_cell_hex.columns = ["Cell_ID","Hexagon_ID","Count"]
            pv_cell_hex["Cell_ID"] = [f"{cell_id}_{dataset_name}" for cell_id in pv_cell_hex["Cell_ID"]]
            nested_dict[dataset_name]["pv_cell_hex"] = pv_cell_hex
            arguments = json.load(open(folder+"arguments.json")) 
            nested_dict[dataset_name]["arguments"] = arguments
        
        tissue_positions_list = data["tissue_positions_list"]
        max_x_range = max(max_x_range,tissue_positions_list["x"].max())*1.1
        max_y_range = max(max_y_range,tissue_positions_list["y"].max())*1.1
        max_col_range = max(max_col_dims,tissue_positions_list["tissue_col"].max())*1.1
        max_row_range = max(max_row_dims,tissue_positions_list["tissue_row"].max())*1.1
        
    return dataset_names, nested_dict, features_df, barcodes_df, max_x_range, max_y_range, max_col_range, max_row_range, scalefactor_hires, scalefactors_for_save, resize_factors

def consensus_features(features_df):
    features_df_all = features_df.copy()
    unique_gene_ids = features_df_all["Gene_ID"].unique()
    features_df_all["consensus_index"] = features_df_all["Gene_ID"].apply(lambda x: np.where(unique_gene_ids==x)[0][0]+1)
    return features_df_all

def adjust_matrix_barcodes(nested_dict, features_df_all, pv_format=False):
    n_barcodes_before = 0
    for dataset_name in nested_dict.keys():
        print(dataset_name)
        matrix = nested_dict[dataset_name]["matrix"]
        features_all = features_df_all[features_df_all["dataset"]==dataset_name]
        for gene_id in matrix["Gene_ID"].unique():
            original_index,consensus_index = features_all[features_all["index_col"]==gene_id][["index_col","consensus_index"]].values[0]
            matrix.loc[matrix["Gene_ID"]==gene_id,"Gene_ID"] = consensus_index
        
        matrix["Barcode_ID"] = matrix["Barcode_ID"] + n_barcodes_before
        nested_dict[dataset_name]["matrix"] = matrix
        if pv_format:
            pv_cell_hex = nested_dict[dataset_name]["pv_cell_hex"]
            pv_cell_hex["Hexagon_ID"] = pv_cell_hex["Hexagon_ID"] + n_barcodes_before
            nested_dict[dataset_name]["pv_cell_hex"] = pv_cell_hex
            
        n_barcodes_before += len(nested_dict[dataset_name]["barcodes"])
    return nested_dict, n_barcodes_before

def stitch_images(nested_dict, grid, max_x_range, max_y_range, max_col_range, max_row_range, scalefactor_hires):
    new_tissue_positions_list = pd.DataFrame(columns=["barcode","in_tissue","tissue_col","tissue_row","x","y"])
    stitched_image = np.zeros((int(max_x_range*scalefactor_hires*grid[0]),int(max_y_range*scalefactor_hires*grid[1]),3))
    number_of_datasets = len(nested_dict.keys())
    for i in range(grid[0]):
        for j in range(grid[1]):

            dataset_idx = i*grid[1]+j
            if dataset_idx >= number_of_datasets:
                break
            print(dataset_idx)
            dataset_names = list(nested_dict.keys())
            tissue_positions_list_to_add = nested_dict[dataset_names[dataset_idx]]["tissue_positions_list"].copy()
            tissue_positions_list_to_add["x"] = tissue_positions_list_to_add["x"] + i*max_x_range
            tissue_positions_list_to_add["y"] = tissue_positions_list_to_add["y"] + j*max_y_range
            tissue_positions_list_to_add["tissue_col"] = tissue_positions_list_to_add["tissue_col"] + i*max_col_range
            tissue_positions_list_to_add["tissue_row"] = tissue_positions_list_to_add["tissue_row"] + j*max_row_range
            new_tissue_positions_list = pd.concat([new_tissue_positions_list,tissue_positions_list_to_add])
            new_tissue_positions_list.reset_index(drop=True,inplace=True)
            
            image_to_add = nested_dict[dataset_names[dataset_idx]]["image"]
            if len(image_to_add.shape) == 2:
                image_to_add = np.stack((image_to_add,)*3,axis=-1)
            image_to_add = np.pad(image_to_add,((0,max(0,int(max_x_range*scalefactor_hires)-image_to_add.shape[0])),(0,max(0,int(max_y_range*scalefactor_hires)-image_to_add.shape[1])),(0,0)))
            image_to_add = image_to_add[0:int(max_x_range*scalefactor_hires),0:int(max_y_range*scalefactor_hires)]
            image_to_add = np.pad(image_to_add,((0,int(max_x_range*scalefactor_hires)-image_to_add.shape[0]),(0,int(max_y_range*scalefactor_hires)-image_to_add.shape[1]),(0,0)))
            stitched_image[i*int(max_x_range*scalefactor_hires):(i+1)*int(max_x_range*scalefactor_hires),j*int(max_y_range*scalefactor_hires):(j+1)*int(max_y_range*scalefactor_hires),:] = image_to_add
    
    new_tissue_positions_list.drop(["dataset","barcode_id"],axis=1,inplace=True)
    return new_tissue_positions_list, stitched_image

def save_output(folderpath, scalefactors_for_save, barcodes_df_all, new_tissue_positions_list, pv_format, nested_dict, dataset_names, features_df_all, stitched_image):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath + '/spatial/')
    
    with open(folderpath +'/spatial/scalefactors_json.json', 'w') as f:
        json.dump(scalefactors_for_save, f)
        
    barcodes_df_all.drop("index_col",axis=1,inplace=True)
    barcodes_df_all.to_csv(folderpath+'/barcodes.tsv',sep='\t',index=False,header=False)
    
    with open(folderpath +'/barcodes.tsv', 'rb') as f_in:
        with gzip.open(folderpath +'/barcodes.tsv.gz', 'wb') as f_out:
            f_out.writelines(f_in)
            
    new_tissue_positions_list.to_csv(folderpath +'/spatial/tissue_positions_list.csv',index=False,header=False)
    
    if pv_format:
        pv_cell_hex = pd.concat([nested_dict[dataset_name]["pv_cell_hex"] for dataset_name in dataset_names])
        pv_cell_hex.to_csv(folderpath +'/spatial/pv_cell_hex.csv',index=False,header=False)
        
    features_df_all = features_df_all.sort_values("consensus_index").drop_duplicates(subset=["Gene_ID"])
    features_df_all["Gene_Name"]=features_df_all["Gene_ID"]
    features_df_all["Type"]="Gene expression"
    features_df_all = features_df_all[["Gene_ID","Gene_Name","Type"]]
    features_df_all.to_csv(folderpath +'/features.tsv',sep='\t',index=False,header=False)
    
    with open(folderpath + '/features.tsv', 'rb') as f_in, gzip.open(folderpath + '/features.tsv.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
    matrix_all = pd.concat([nested_dict[dataset_name]["matrix"] for dataset_name in dataset_names])
    metadata = f'%%MatrixMarket matrix coordinate integer general\n%metadata_json: {{"software_version": "Pseudovisium", "format_version": 1}}\n{len(features_df_all)} {barcodes_df_all.shape[0]} {len(matrix_all)}'
    matrix_all.to_csv(folderpath +'/matrix.mtx',index=False,header=False,sep=" ")
    with open(folderpath + '/matrix.mtx', 'r') as original: data = original.read()
    with open(folderpath + '/matrix.mtx', 'w') as modified: modified.write(metadata + '\n' + data)
    with open(folderpath + '/matrix.mtx', 'rb') as f_in, gzip.open(folderpath + '/matrix.mtx.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
    plt.imsave(folderpath +'/spatial/tissue_hires_image.png', stitched_image)
    lowres_image = cv2.resize(stitched_image,(int(stitched_image.shape[1]/10),int(stitched_image.shape[0]/10)))
    plt.imsave(folderpath +'/spatial/tissue_lowres_image.png', lowres_image)

def merge_visium(folders, output_path, project_name, pv_format=False):
    dataset_names, nested_dict, features_df, barcodes_df, max_x_range, max_y_range, max_col_range, max_row_range, scalefactor_hires, scalefactors_for_save, resize_factors = merge_data(folders, pv_format)
    n_rows = max(3,int(np.sqrt(len(folders))))
    n_cols = int(np.ceil(len(folders)/n_rows))
    grid = [n_rows, n_cols]
    features_df_all = consensus_features(features_df)
    nested_dict, n_barcodes_before = adjust_matrix_barcodes(nested_dict, features_df_all, pv_format)
    barcodes_df_all = barcodes_df.copy()
    barcodes_df_all.reset_index(drop=True,inplace=True)
    barcodes_df_all["index_col"] = barcodes_df_all.index + 1
    barcodes_df_all.drop("dataset",axis=1,inplace=True)
    new_tissue_positions_list, stitched_image = stitch_images(nested_dict, grid, max_x_range, max_y_range,  max_col_range, max_row_range,scalefactor_hires)
    folderpath = output_path+ '/pseudovisium/' + project_name
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
    save_output(folderpath, scalefactors_for_save, barcodes_df_all, new_tissue_positions_list, pv_format, nested_dict, dataset_names, features_df_all, stitched_image)

def main():
    parser = argparse.ArgumentParser(description="Merge Pseudovisium/Visium format files.")
    parser.add_argument("--folders", "-f", nargs="+", help="List of folders containing Pseudovisium/Visium output", required=True)
    parser.add_argument("--output_path", "-o", default="/Users/k23030440/", help="Output folder path")
    parser.add_argument("--project_name", "-p", default="visium_merged", help="Project name for output")
    parser.add_argument("--pv_format", action="store_true", help="Indicate if input is in Pseudovisium format")
    args = parser.parse_args()

    folders = [folder+"/" if not folder.endswith("/") else folder for folder in args.folders]
    merge_visium(folders, args.output_path, args.project_name, args.pv_format)

if __name__ == "__main__":
    main()
