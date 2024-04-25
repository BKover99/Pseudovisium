import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import pandas as pd
from pysal.lib import weights
from pysal.explore import esda
import geopandas as gpd
import seaborn as sns
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import datetime
import base64
from shapely.geometry import Point
from libpysal import weights
from esda import Moran
from adjustText import adjust_text
from io import BytesIO
import base64



def generate_qc_report(folders, output_folder="/Users/k23030440/", gene_names=["RYR3", "AQP4", "THBS1"], include_morans_i=False,max_workers=4,normalisation=False):
    #if any entry in folders lacks final /, then add
    folders = [folder if folder[-1]=="/" else folder + "/" for folder in folders]
    #same with output_folder
    output_folder = output_folder if output_folder[-1]=="/" else output_folder + "/"

    replicates_data = []
    for folder in tqdm(folders, desc="Processing folders"):
        # Extract the dataset name from the folder path
        dataset_name = folder.split("/")[-2]

        # Load files
        tissue_positions_list = pd.read_csv(folder + "spatial/tissue_positions_list.csv", header=None)
        tissue_positions_list.columns = ["barcode", "in_tissue", "tissue_col", "tissue_row", "x", "y"]
        matrix = pd.read_csv(folder + "matrix.mtx", header=3, sep=" ")
        matrix.columns = ["Gene_ID", "Barcode_ID", "Counts"]
        features = pd.read_csv(folder + "features.tsv", header=None, sep="\t")
        features.columns = ["Gene_ID", "Gene_Name", "Type"]
        features["index_col"] = features.index + 1
        barcodes = pd.read_csv(folder + "barcodes.tsv", header=None, sep="\t")
        barcodes.columns = ["Barcode_ID"]
        barcodes["index_col"] = barcodes.index + 1

        arguments = json.load(open(folder + "arguments.json"))
        #if cell_id_colname is not NA, then we have cell info
        cell_info= True if arguments["cell_id_colname"]!="NA" else False
        quality_per_hexagon = arguments["quality_per_hexagon"]
        quality_per_probe = arguments["quality_per_probe"]

        if quality_per_hexagon:
            hexagon_quality = pd.read_csv(folder + "spatial/quality_per_hexagon.csv", header=None)
            #name cols as barcode, quality, count
            hexagon_quality.columns = ["Hexagon_ID", "Quality", "Count"]
            #add dataset name
            hexagon_quality["Dataset"] = dataset_name
            hexagons_above_100 = hexagon_quality[hexagon_quality["Count"]>100]
            hexagons_q_below_20 = hexagons_above_100[hexagons_above_100["Quality"]<20]
            pct_hexagons_q_below_20 = len(hexagons_q_below_20)/len(hexagons_above_100)
            

        if quality_per_probe:
            probe_quality = pd.read_csv(folder + "spatial/quality_per_probe.csv", header=None)
            #name cols as barcode, quality, count
            probe_quality.columns = ["Probe_ID", "Quality", "Count"]
            probe_quality["Dataset"] = dataset_name
            non_ctrl_probes = probe_quality[~probe_quality["Probe_ID"].str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]
            non_ctrl_probes_q_below_20 = non_ctrl_probes[non_ctrl_probes["Quality"]<20]
            pct_non_ctrl_probes_q_below_20 = len(non_ctrl_probes_q_below_20)/len(non_ctrl_probes)
            
        
        if cell_info:
            pv_cell_hex = pd.read_csv(folder + "spatial/pv_cell_hex.csv", header=None)
            pv_cell_hex.columns = ["Cell_ID", "Hexagon_ID", "Count"]
            pv_cell_hex["Hexagon_ID"] = pv_cell_hex["Hexagon_ID"] + 1




        arguments = json.load(open(folder + "arguments.json"))
        hexagon_size = arguments["hexagon_size"]
        image_pixels_per_um = arguments["image_pixels_per_um"]

        tissue_positions_list = tissue_positions_list[tissue_positions_list["in_tissue"] == 1]
        tissue_positions_list["barcode_id"] = [barcodes[barcodes["Barcode_ID"] == barcode].index_col.values[0] for barcode in tissue_positions_list["barcode"]]
        in_tissue_barcodes = tissue_positions_list["barcode"].index + 1

        n_barcodes_per_tissue = len(tissue_positions_list)

        # Join the x and y columns from tissue_positions_list based on Barcode_ID and barcode_id
        matrix_joined = pd.merge(matrix, tissue_positions_list[["barcode_id", "x", "y"]], left_on="Barcode_ID", right_on="barcode_id")
        matrix_joined = pd.merge(matrix_joined, features[["Gene_ID", "index_col"]], left_on="Gene_ID", right_on="index_col")

        if quality_per_hexagon:
            #in hexagon_quality keep only rows where hexagon_id is in index_col
            hexagon_quality = hexagon_quality[hexagon_quality["Hexagon_ID"].isin(matrix_joined["barcode_id"])]
            matrix_joined = pd.merge(matrix_joined, hexagon_quality, left_on="barcode_id", right_on="Hexagon_ID")

        # Calculate key metrics
        grouped_matrix = matrix_joined.groupby("Barcode_ID")["Counts"].sum()
        number_of_hex_above_100 = np.sum(grouped_matrix > 50)

        grouped_matrix = matrix_joined.groupby("Gene_ID_y")["Barcode_ID"].count()
        pct5_plex = np.sum(grouped_matrix > 0.05 * n_barcodes_per_tissue)
        probe_names = grouped_matrix[grouped_matrix > 0.05 * n_barcodes_per_tissue].index.values

        grouped_matrix = matrix_joined.groupby("Barcode_ID")["Counts"].sum()
        median_counts = np.median(grouped_matrix)
        cv_counts = np.std(grouped_matrix) / np.mean(grouped_matrix)

        grouped_matrix = matrix_joined.groupby("Barcode_ID")["Gene_ID_y"].count()
        median_features = np.median(grouped_matrix)
        cv_features = np.std(grouped_matrix) / np.mean(grouped_matrix)

        number_of_probes = len(features)
        number_of_genes = len(features[~features["Gene_ID"].str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")])
        

        neg_control_probes = features[features["Gene_ID"].str.contains("Probe")].index + 1
        neg_control_counts = np.sum(matrix[matrix["Gene_ID"].isin(neg_control_probes)]["Counts"])
        total_counts = np.sum(matrix["Counts"])
        prop_neg_control = neg_control_counts / total_counts
        if cell_info:
            pv_cell_hex_assigned = pv_cell_hex[(pv_cell_hex["Cell_ID"] != "UNASSIGNED") & (pv_cell_hex["Cell_ID"] != -1)]
            grouped_pv_cell_hex_assigned = pv_cell_hex_assigned.groupby("Hexagon_ID")["Cell_ID"].count()
            median_cells_per_hex = np.median(grouped_pv_cell_hex_assigned)

            counts_per_cell = pv_cell_hex_assigned.groupby("Cell_ID")["Count"].sum()
            median_counts_per_cell = np.median(counts_per_cell)

            pv_cell_hex_unassigned = pv_cell_hex[(pv_cell_hex["Cell_ID"] != "UNASSIGNED") | (pv_cell_hex["Cell_ID"] != -1)]
            #merge with grouped_pv_cell_hex_assigned
            grouped_pv_cell_hex_assigned_sum = pv_cell_hex_assigned.groupby("Hexagon_ID")["Count"].sum()
            merged_assigned_unassigned = pd.merge(grouped_pv_cell_hex_assigned_sum,pv_cell_hex_unassigned,left_on="Hexagon_ID",right_on="Hexagon_ID",how="inner")
            merged_assigned_unassigned = merged_assigned_unassigned.rename(columns={"Count_x":"Assigned_count","Count_y":"Unassigned_count"})
            merged_assigned_unassigned = merged_assigned_unassigned.fillna(0)
            #add pct_unassigned
            merged_assigned_unassigned["pct_unassigned"] = merged_assigned_unassigned["Unassigned_count"]/(merged_assigned_unassigned["Assigned_count"]+merged_assigned_unassigned["Unassigned_count"])
            #merge with matrix
            merged_assigned_unassigned = pd.merge(merged_assigned_unassigned,tissue_positions_list,left_on="Hexagon_ID",right_on="barcode_id",how="inner",suffixes=('_a', '_b'))
            median_unassigned_pct = np.median(merged_assigned_unassigned["pct_unassigned"])


            # merge tissue_positions_list with pv_cell_hex
            tissue_positions_pv_cell_hex = pd.merge(tissue_positions_list, pv_cell_hex, left_on="barcode_id", right_on="Hexagon_ID", how="inner")
            #sum up the number of Cell_ID in every Hexagon_ID but keep x and y intact
            tissue_positions_pv_cell_hex_sum = tissue_positions_pv_cell_hex.groupby(["x","y"]).agg({"Cell_ID":"count","Hexagon_ID":"first"}).reset_index()
            #remove multi level index
            tissue_positions_pv_cell_hex_sum.reset_index(inplace=True)
            #rename Cell_ID to count
            tissue_positions_pv_cell_hex_sum.rename(columns={"Cell_ID":"counts"},inplace=True)
            density_cv = np.std(tissue_positions_pv_cell_hex_sum["counts"])/np.mean(tissue_positions_pv_cell_hex_sum["counts"])
            density_morans_i = get_morans_i("density", tissue_positions_pv_cell_hex_sum, tissue_positions_list,max_workers=max_workers) 
            




        plot_df = not_working_probe_based_on_sum(matrix_joined, sample_id=dataset_name)
        not_working_probes = plot_df[plot_df["Probe category"]=="Bad"].index.values
        n_probes_not_working = len(not_working_probes)
        
        

        replicate_data = {
            "dataset_name": dataset_name,
            "metrics_table_data": {
                "Number of hexagons with at least 100 counts": int(number_of_hex_above_100),
                "Number of genes in at least 5% of hexagons": int(pct5_plex),
                "Median counts per hexagon": int(median_counts),
                "Median features per hexagon": int(median_features),
                "Total number of probes (inc. ctrl)": int(number_of_probes),
                "Number of genes": int(number_of_genes),
                "Proportion of neg_control probes": np.round(prop_neg_control, 5),
                "Number of bad probes (Sum)": n_probes_not_working,
                "Features CV": np.round(cv_features, 5),
                "Counts CV": np.round(cv_counts, 5),
                "Features Morans I": np.round(get_morans_i("features", matrix_joined, tissue_positions_list,max_workers=max_workers),5),
                "Counts Morans I": np.round(get_morans_i("counts", matrix_joined, tissue_positions_list,max_workers=max_workers),5)
            },
            "matrix_joined": matrix_joined,
            "features": features,
            "tissue_positions_list": tissue_positions_list,
            "hexagon_size": hexagon_size,
            "image_pixels_per_um": image_pixels_per_um,
            "barcodes": barcodes,
            "probe_sum_stripplot_df":plot_df
        }

        if include_morans_i:
            replicate_data["morans_i"] = get_morans_i("all", matrix_joined, tissue_positions_list,max_workers=max_workers)
            plot_df_morans_i = not_working_probe_based_on_morans_i(replicate_data["morans_i"], sample_id=dataset_name)
            not_working_probes = plot_df_morans_i[plot_df_morans_i["Probe category"]=="Bad"].index.values
            n_probes_not_working = len(not_working_probes)
            #add to replicate_data metrics_table_data
            replicate_data["metrics_table_data"]["Number of bad probes (Morans I)"] = n_probes_not_working
            replicate_data["morans_i_stripplot_df"] = plot_df_morans_i

        if cell_info:
            replicate_data["metrics_table_data"]["Median density (cells per hexagon)"] = int(median_cells_per_hex)
            replicate_data["metrics_table_data"]["Median counts per cell"] = int(median_counts_per_cell)
            replicate_data["metrics_table_data"]["Median pct unassigned"] = np.round(median_unassigned_pct, 5)
            replicate_data["merged_assigned_unassigned"] = merged_assigned_unassigned
            replicate_data["cell_density_df"] = tissue_positions_pv_cell_hex_sum

            replicate_data["metrics_table_data"]["Density (cells per hexagon) CV"] = np.round(density_cv, 5)
            replicate_data["metrics_table_data"]["Density (cells per hexagon) Morans I"] = np.round(density_morans_i, 5)

        if quality_per_hexagon:
            replicate_data["metrics_table_data"]["Pct hexagons with quality below 20"] = np.round(pct_hexagons_q_below_20, 5)
            replicate_data["hexagon_quality"] = hexagon_quality
            replicate_data["metrics_table_data"]["Quality Morans I"] = np.round(get_morans_i("Quality", matrix_joined, tissue_positions_list,max_workers=max_workers),5)
            replicate_data["metrics_table_data"]["Median hexagon quality"] = np.round(hexagon_quality["Quality"].median(), 5)


        if quality_per_probe:
            replicate_data["metrics_table_data"]["Pct non-ctrl probes with quality below 20"] = np.round(pct_non_ctrl_probes_q_below_20, 5)
            replicate_data["probe_quality"] = probe_quality
            plot_df_quality_per_probe = not_working_probe_based_on_quality(probe_quality, sample_id=dataset_name)
            replicate_data["probe_quality_stripplot_df"] = plot_df_quality_per_probe

        replicates_data.append(replicate_data)

    html_code = generate_dashboard_html(replicates_data, gene_names, include_morans_i,quality_per_hexagon,quality_per_probe,cell_info,normalisation=normalisation)

    # Save HTML code to a file
    with open(output_folder + "metrics_dashboard.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_code)
        print("HTML file generated successfully!")
    
    #in the same output folder generate a folder called pv_qc_ date
    data_output_folder = output_folder + "pv_qc_" + str(datetime.datetime.now().date())
    #if that dir exists_ add a number to the end
    i=1
    while os.path.exists(data_output_folder):
        data_output_folder = output_folder + "pv_qc_" + str(datetime.datetime.now().date()) + "_" + str(i)
        i+=1
    os.mkdir(data_output_folder)

    #save the metrics table as a csv
    metrics_table = pd.DataFrame([replicate_data["metrics_table_data"] for replicate_data in replicates_data])
    #add a column for the dataset name
    metrics_table["Dataset"] = [replicate_data["dataset_name"] for replicate_data in replicates_data]
    metrics_table.to_csv(data_output_folder + "/metrics_table.csv", index=False)

    #save probe_quality as a csv
    if quality_per_probe:
        probe_quality = pd.concat([replicate_data["probe_quality_stripplot_df"] for replicate_data in replicates_data])
        probe_quality.to_csv(data_output_folder + "/probe_quality.csv", index=False)

    #save morans i values for each gene
    if include_morans_i:
        morans_i = pd.concat([replicate_data["morans_i_stripplot_df"] for replicate_data in replicates_data])
        morans_i.to_csv(data_output_folder + "/morans_i.csv", index=False)

    #save sum of probes stripplot
    sum_stripplot = pd.concat([replicate_data["probe_sum_stripplot_df"] for replicate_data in replicates_data])
    sum_stripplot.to_csv(data_output_folder + "/sum_stripplot.csv", index=False)



def generate_dashboard_html(replicates_data, gene_names, include_morans_i,quality_per_hexagon,quality_per_probe,cell_info,normalisation=False):
    metrics_html = """
        <div id="metric-details">
            <h2>Counts Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
    """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <th>{replicate_data['dataset_name']}</th>
        """
    metrics_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Median counts per hexagon</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Median counts per hexagon']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Number of hexagons with at least 100 counts</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Number of hexagons with at least 100 counts']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Proportion of neg_control probes</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Proportion of neg_control probes']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Counts Morans I</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Counts Morans I']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Counts CV</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Counts CV']}</td>
        """
    metrics_html += """
                    </tr>
                </tbody>
            </table>

            <h2>Features Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
    """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <th>{replicate_data['dataset_name']}</th>
        """
    metrics_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total number of probes (inc. ctrl)</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Total number of probes (inc. ctrl)']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Number of genes</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Number of genes']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Median features per hexagon</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Median features per hexagon']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Number of genes in at least 5% of hexagons</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Number of genes in at least 5% of hexagons']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Features Morans I</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Features Morans I']}</td>
        """
    metrics_html += """
                    </tr>
                    <tr>
                        <td>Features CV</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Features CV']}</td>
        """
    metrics_html += """
                    </tr>
                </tbody>
            </table>

            <h2>Uninformative Probes Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
    """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <th>{replicate_data['dataset_name']}</th>
        """
    metrics_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Number of bad probes (Sum)</td>
        """
    for replicate_data in replicates_data:
        metrics_html += f"""
                        <td>{replicate_data['metrics_table_data']['Number of bad probes (Sum)']}</td>
        """
    metrics_html += """
                    </tr>
    """
    if include_morans_i:
        metrics_html += """
                    <tr>
                        <td>Number of bad probes (Morans I)</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Number of bad probes (Morans I)', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
        """
    metrics_html += """
                </tbody>
            </table>
    """

    if cell_info:
        metrics_html += """
            <h2>Cell Info Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <th>{replicate_data['dataset_name']}</th>
            """
        metrics_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Median counts per cell</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Median counts per cell', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Median pct unassigned</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Median pct unassigned', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Median density (cells per hexagon)</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Median density (cells per hexagon)', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Density (cells per hexagon) CV</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Density (cells per hexagon) CV', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Density (cells per hexagon) Morans I</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Density (cells per hexagon) Morans I', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                </tbody>
            </table>
        """

    if quality_per_hexagon:
        metrics_html += """
            <h2>Quality Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <th>{replicate_data['dataset_name']}</th>
            """
        metrics_html += """
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Median hexagon quality</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Median hexagon quality', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Quality Morans I</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Quality Morans I', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
                    <tr>
                        <td>Pct hexagons with quality below 20</td>
        """
        for replicate_data in replicates_data:
            metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Pct hexagons with quality below 20', 'N/A')}</td>
            """
        metrics_html += """
                    </tr>
        """
        if quality_per_probe:
            metrics_html += """
                    <tr>
                        <td>Pct non-ctrl probes with quality below 20</td>
            """
            for replicate_data in replicates_data:
                metrics_html += f"""
                        <td>{replicate_data['metrics_table_data'].get('Pct non-ctrl probes with quality below 20', 'N/A')}</td>
                """
            metrics_html += """
                    </tr>
            """
        metrics_html += """
                </tbody>
            </table>
        """

    metrics_html += """
        </div>
    """

    hexagon_plots_html = ""
    for i, gene_name in enumerate(gene_names):
        if i % 3 == 0:
            hexagon_plots_html += f"""
            <div class="row">
            """
        gene_found = False
        for replicate_data in replicates_data:
            if gene_name in replicate_data['features']['Gene_Name'].tolist():
                gene_found = True
                hexagon_df = get_df_for_gene(replicate_data['matrix_joined'], replicate_data['tissue_positions_list'], gene_name,normalisation)
                hexagon_html = hexagon_plot_to_html(hexagon_df, replicate_data['hexagon_size'], replicate_data['image_pixels_per_um'], gene_name, replicate_data['dataset_name'])
                hexagon_plots_html += f"""
                <div class="col">
                    {hexagon_html}
                </div>
                """
            else:
                hexagon_plots_html += """
                <div class="col">
                    <p>Gene not found in dataset.</p>
                </div>
                """
        if gene_found:
            hexagon_plots_html += """
            <div class="col">
                <div id="colorbar"></div>
            </div>
            """
        if (i + 1) % 3 == 0 or i == len(gene_names) - 1:
            hexagon_plots_html += """
            </div>
            """

    # Section for nFeature Hexagon plots
    nfeature_hexagon_plots_html = ""
    for i, replicate_data in enumerate(replicates_data):
        if i % 3 == 0:
            nfeature_hexagon_plots_html += f"""
            <div class="row">
            """
        unique_features_per_hexagon = get_unique_features_per_hexagon(replicate_data['matrix_joined'])
        hexagon_html = hexagon_plot_to_html(unique_features_per_hexagon, replicate_data['hexagon_size'], replicate_data['image_pixels_per_um'], "nFeature", replicate_data['dataset_name'],replicate_data['metrics_table_data']["Features Morans I"])
        nfeature_hexagon_plots_html += f"""
        <div class="col">
            <h3>{replicate_data['dataset_name']}</h3>
            {hexagon_html}
        </div>
        """
        if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
            nfeature_hexagon_plots_html += """
            </div>
            """

    if cell_info:
        cell_density_hexagon_plots_html = ""
        for i, replicate_data in enumerate(replicates_data):
            if i % 3 == 0:
                cell_density_hexagon_plots_html += f"""
                <div class="row">
                """
            cell_density_df= replicate_data["cell_density_df"]
            hexagon_html = hexagon_plot_to_html(cell_density_df, replicate_data['hexagon_size'], replicate_data['image_pixels_per_um'], "Density", replicate_data['dataset_name'],replicate_data['metrics_table_data']["Density (cells per hexagon) Morans I"])
            cell_density_hexagon_plots_html += f"""
            <div class="col">
                <h3>{replicate_data['dataset_name']}</h3>
                {hexagon_html}
            </div>
            """
            if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
                cell_density_hexagon_plots_html += """
                </div>
                """




    if quality_per_hexagon:
        quality_hexagon_plots_html = ""
        for i, replicate_data in enumerate(replicates_data):
            if i % 3 == 0:
                quality_hexagon_plots_html += f"""
                <div class="row">
                """
            hexagon_quality = get_quality_per_hexagon(replicate_data['matrix_joined'])
            hexagon_html = hexagon_plot_to_html(hexagon_quality, replicate_data['hexagon_size'], replicate_data['image_pixels_per_um'], "Quality", replicate_data['dataset_name'],replicate_data['metrics_table_data']["Quality Morans I"])
            quality_hexagon_plots_html += f"""
            <div class="col">
                <h3>{replicate_data['dataset_name']}</h3>
                {hexagon_html}
            </div>
            """
            if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
                quality_hexagon_plots_html += """
                </div>
                """

                

    total_hexagon_plots_html = ""
    for i, replicate_data in enumerate(replicates_data):
        if i % 3 == 0:
            total_hexagon_plots_html += f"""
            <div class="row">
            """
        total_counts_per_hexagon = get_total_counts_per_hexagon(replicate_data['matrix_joined'])
        hexagon_html = hexagon_plot_to_html(total_counts_per_hexagon, replicate_data['hexagon_size'], replicate_data['image_pixels_per_um'], "Total exp", replicate_data['dataset_name'],replicate_data['metrics_table_data']["Counts Morans I"])
        total_hexagon_plots_html += f"""
        <div class="col">
            <h3>{replicate_data['dataset_name']}</h3>
            {hexagon_html}
        </div>
        """
        if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
            total_hexagon_plots_html += """
            </div>
            """

    sums_comparison_html = ""
    pairs = [(i, j) for i in range(len(replicates_data)) for j in range(i + 1, len(replicates_data))]
    for i, (index1, index2) in enumerate(pairs):
        if i % 3 == 0:
            sums_comparison_html += f"""
            <div class="row">
            """
        sums1 = get_probe_sums(replicates_data[index1]['matrix_joined'])
        sums2 = get_probe_sums(replicates_data[index2]['matrix_joined'])
        sums_plot_html = plot_sums_to_html(sums1, sums2, replicates_data[index1]['dataset_name'], replicates_data[index2]['dataset_name'])
        sums_comparison_html += f"""
        <div class="col">
            <h3>{replicates_data[index1]['dataset_name']} vs {replicates_data[index2]['dataset_name']}</h3>
            {sums_plot_html}
        </div>
        """
        if (i + 1) % 3 == 0 or i == len(pairs) - 1:
            sums_comparison_html += """
            </div>
            """

    abundance_correlation_heatmap_html = plot_abundance_correlation_heatmap(replicates_data)

    morans_i_comparison_html = ""
    morans_i_heatmap_html = ""
    morans_i_stripplot_html = ""
    if include_morans_i:
        pairs = [(i, j) for i in range(len(replicates_data)) for j in range(i + 1, len(replicates_data))]
        for i, (index1, index2) in enumerate(pairs):
            if i % 3 == 0:
                morans_i_comparison_html += f"""
                <div class="row">
                """
            morans_i1 = replicates_data[index1]['morans_i']
            morans_i2 = replicates_data[index2]['morans_i']
            morans_i_plot_html = plot_morans_i_to_html(morans_i1, morans_i2, replicates_data[index1]['dataset_name'], replicates_data[index2]['dataset_name'])
            morans_i_comparison_html += f"""
            <div class="col">
                <h3>{replicates_data[index1]['dataset_name']} vs {replicates_data[index2]['dataset_name']}</h3>
                {morans_i_plot_html}
            </div>
            """
            if (i + 1) % 6 == 0 or i == len(pairs) - 1:
                morans_i_comparison_html += """
                </div>
                """
        morans_i_heatmap_html = plot_morans_i_correlation_heatmap(replicates_data)
        for i, replicate_data in enumerate(replicates_data):
            if i % 3 == 0:
                morans_i_stripplot_html += f"""
                <div class="row">
                """
            morans_i_stripplot_df = replicate_data['morans_i_stripplot_df']
            morans_i_stripplot_html += f"""
            <div class="col">
                <h3>{replicate_data['dataset_name']}</h3>
                {probe_stripplot(morans_i_stripplot_df, sample_id=replicate_data['dataset_name'],legend=True, col_to_plot="Morans_I")}
            </div>
            """
            if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
                morans_i_stripplot_html += """
                </div>
                """

    sums_i_stripplot_html = ""
    for i, replicate_data in enumerate(replicates_data):
        if i % 3 == 0:
            sums_i_stripplot_html += f"""
            <div class="row">
            """
        plot_df = replicate_data['probe_sum_stripplot_df']
        sums_i_stripplot_html += f"""
        <div class="col">
            <h3>{replicate_data['dataset_name']}</h3>
            {probe_stripplot(plot_df, sample_id=replicate_data['dataset_name'],legend=True, col_to_plot="log_counts")}
        </div>
        """
        if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
            sums_i_stripplot_html += """
            </div>
            """

    quality_stripplot_html = ""
    if quality_per_probe:
        for i, replicate_data in enumerate(replicates_data):
            if i % 3 == 0:
                quality_stripplot_html += f"""
                <div class="row">
                """
            plot_df = replicate_data['probe_quality_stripplot_df']
            quality_stripplot_html += f"""
            <div class="col">
                <h3>{replicate_data['dataset_name']}</h3>
                {probe_stripplot(plot_df, sample_id=replicate_data['dataset_name'],legend=True, col_to_plot="Quality")}
            </div>
            """
            if (i + 1) % 3 == 0 or i == len(replicates_data) - 1:
                quality_stripplot_html += """
                </div>
                """

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pseudovisium QC</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 12px;
            }}
            th, td {{
                padding: 6px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                white-space: nowrap;
            }}
            .row {{
                display: flex;
                flex-wrap: wrap;
                margin: 0 -20px;
            }}
            .col {{
                flex: 0 0 22%;
                max-width: 22%;
                padding: 0 20px;
                margin: 0 10px;
            }}
            .plot-container {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">Pseudovisium QC</h1>
        <p style="text-align:center;">Written by Bence Kover (2024)</p>
        <div class="container">
            <div class="dropdown">
                <label for="metrics-select">Select Metric:</label>
                <select id="metrics-select">
                    <option value="table">Table of Key Metrics</option>
                    <option value="abundance-correlation-heatmap">Abundance Correlation Heatmap</option>
                    <option value="sums-comparison">Abundance Comparison</option>
                    <option value="sums-i-stripplot">Abundance Stripplot</option>
                    {'<option value="morans-i-heatmap">Morans I Correlation Heatmap</option>' if include_morans_i else ""}
                    {'<option value="morans-i-comparison">Morans I Comparison</option>' if include_morans_i else ""}
                    {'<option value="morans-i-stripplot">Morans I Stripplot</option>' if include_morans_i else ""}
                    <option value="plot">Hexagon Plots for Genes of Interest</option>
                    <option value="nfeature_hexagon_plots">Number of Features per Hexagon Plots</option>
                    <option value="total_hexagon_plots">Total Hexagon Plots</option>
                    {'<option value="cell_density_hexagon_plots">Cell Density Hexagon Plots</option>' if cell_info else ""}
                    {'<option value="quality_hexagon_plots">Quality Hexagon Plots</option>' if quality_per_hexagon else ""}
                    {'<option value="probe_quality_stripplot">Probe Quality Stripplot</option>' if quality_per_probe else ""}
                </select>
            </div>
            <div id="metric-details-container">
                {metrics_html}
            </div>
            <div id="plot" class="plot-container">
                {hexagon_plots_html}
            </div>
            <div id="sums-comparison" class="plot-container">
                {sums_comparison_html}
            </div>
            <div id="abundance-correlation-heatmap" class="plot-container">
                {abundance_correlation_heatmap_html}
            </div>
            <div id="morans-i-heatmap" class="plot-container">
                {morans_i_heatmap_html}
            </div>
            <div id="morans-i-stripplot" class="plot-container">
                {morans_i_stripplot_html}
            </div>
            <div id="sums-i-stripplot" class="plot-container">
                {sums_i_stripplot_html}
            </div>
            <div id="nfeature_hexagon_plots" class="plot-container">
                {nfeature_hexagon_plots_html}
            </div>
            <div id="quality_hexagon_plots" class="plot-container">
                {quality_hexagon_plots_html}
            </div>
            <div id="total_hexagon_plots" class="plot-container">
                {total_hexagon_plots_html}
            </div>
            <div id="morans-i-comparison" class="plot-container">
                {morans_i_comparison_html}
            </div>
            <div id="probe_quality_stripplot" class="plot-container">
                {quality_stripplot_html}
            </div>
            <div id="cell_density_hexagon_plots" class="plot-container">
                {cell_density_hexagon_plots_html}
            </div>
        </div>

        <script>
            const select = document.getElementById('metrics-select');
            const metricDetailsContainer = document.getElementById('metric-details-container');
            const plotContainer = document.getElementById('plot');
            const sumsComparisonContainer = document.getElementById('sums-comparison');
            const abundanceCorrelationHeatmapContainer = document.getElementById('abundance-correlation-heatmap');
            const moransIHeatmapContainer = document.getElementById('morans-i-heatmap');
            const moransIStripplotContainer = document.getElementById('morans-i-stripplot');
            const sumsIStripplotContainer = document.getElementById('sums-i-stripplot');
            const nfeatureHexagonContainer = document.getElementById('nfeature_hexagon_plots');
            const qualityHexagonContainer = document.getElementById('quality_hexagon_plots');
            const totalHexagonContainer = document.getElementById('total_hexagon_plots');
            const moransIComparisonContainer = document.getElementById('morans-i-comparison');
            const probeQualityStripplotContainer = document.getElementById('probe_quality_stripplot');
            const cellDensityHexagonContainer = document.getElementById('cell_density_hexagon_plots');


            function updateMetricDetails() {{
                const selectedMetric = select.value;
                const plotContainers = document.querySelectorAll('.plot-container');

                // Hide all plot containers
                plotContainers.forEach(container => {{
                    container.style.display = 'none';
                }});

                // Hide the metric details container
                metricDetailsContainer.style.display = 'none';

                if (selectedMetric === 'table') {{
                    metricDetailsContainer.style.display = 'block';
                }} else if (selectedMetric === 'plot') {{
                    plotContainer.style.display = 'block';
                }} else if (selectedMetric === 'sums-comparison') {{
                    sumsComparisonContainer.style.display = 'block';
                }} else if (selectedMetric === 'abundance-correlation-heatmap') {{
                    abundanceCorrelationHeatmapContainer.style.display = 'block';
                }} else if (selectedMetric === 'morans-i-heatmap') {{
                    moransIHeatmapContainer.style.display = 'block';
                }} else if (selectedMetric === 'morans-i-stripplot') {{
                    moransIStripplotContainer.style.display = 'block';
                }} else if (selectedMetric === 'sums-i-stripplot') {{
                    sumsIStripplotContainer.style.display = 'block';
                }} else if (selectedMetric === 'nfeature_hexagon_plots') {{
                    nfeatureHexagonContainer.style.display = 'block';
                }} else if (selectedMetric === 'quality_hexagon_plots') {{
                    qualityHexagonContainer.style.display = 'block';
                }} else if (selectedMetric === 'total_hexagon_plots') {{
                    totalHexagonContainer.style.display = 'block';
                }} else if (selectedMetric === 'morans-i-comparison') {{
                    moransIComparisonContainer.style.display = 'block';
                }}
                else if (selectedMetric === 'probe_quality_stripplot') {{
                    probeQualityStripplotContainer.style.display = 'block';
                }}
                else if (selectedMetric === 'cell_density_hexagon_plots') {{
                    cellDensityHexagonContainer.style.display = 'block';
                }}

            }}

            select.addEventListener('change', updateMetricDetails);

            // Initial update
            updateMetricDetails();
        </script>
    </body>
    </html>
    """
    return html_code


def not_working_probe_based_on_sum(matrix_joined,sample_id="Sample1"):
    grouped_matrix = matrix_joined.groupby("Gene_ID_y")["Counts"].sum()
    #where index has control|blank|Control|Blank|BLANK in it
    grouped_matrix_neg_probes = grouped_matrix[grouped_matrix.index.str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]
    grouped_matrix_true_probes = grouped_matrix[~grouped_matrix.index.str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]     

    #create a plot_df that is grouped_matrix and a column specifying whether the gene is a neg control or not
    plot_df = pd.DataFrame(grouped_matrix)
    plot_df["Probe category"] = [1 if gene in grouped_matrix_neg_probes.index.values else 0 for gene in plot_df.index.values]
    plot_df["gene"] = plot_df.index.values
    plot_df["log_counts"] = np.log10(plot_df["Counts"])

    #iterate through the true probes and see whether they are significantly outside of the distribution of the neg probes
    for gene in grouped_matrix_true_probes.index.values:
        gene_count = plot_df.loc[gene,"log_counts"]
        neg_probes_count = plot_df[plot_df["Probe category"]==1]["log_counts"]
        #get pval with ttest
        mean = np.mean(neg_probes_count)
        std = np.std(neg_probes_count)
        p_val = stats.norm.cdf(gene_count, loc=mean, scale=std)
        p_val = 1-p_val
        if p_val*len(grouped_matrix_true_probes)<0.05:
            plot_df.loc[gene,"Probe category"] = 2
        
    plot_df["Probe category"] = ["Neg_control" if x==1 else "Bad" if x==0 else "Good" for x in plot_df["Probe category"]]
    plot_df["Sample"] = sample_id
    #order based on probe category
    plot_df = plot_df.sort_values("Probe category")
    return plot_df






def probe_stripplot(plot_df, col_to_plot="log_counts", sample_id="Sample 1", legend=False):
    fig, ax = plt.subplots(figsize=(2, 2.5))
    # Define jitter amount
    jitter = 0.1

    # Create a dictionary to store x and y values for each point
    points = {"x": [], "y": [], "cat": [], "index": []}

    # Assign x and y values with jitter for each point
    for index, row in plot_df.iterrows():
        cat = row["Probe category"]
        x = 0 + np.random.uniform(-jitter, jitter)
        y = row[col_to_plot]
        points["x"].append(x)
        points["y"].append(y)
        points["cat"].append(cat)
        points["index"].append(index)

    # Plot the points for each category
    for cat in [ "Good", "Bad","Neg_control"]:
        mask = [c == cat for c in points["cat"]]
        ax.scatter([x for x, m in zip(points["x"], mask) if m],
                   [y for y, m in zip(points["y"], mask) if m],
                   alpha=0.8, label=cat)

    # Draw a straight line at the mean of the neg controls
    neg_control_mean = np.mean(plot_df[plot_df["Probe category"] == "Neg_control"][col_to_plot])
    ax.axhline(neg_control_mean, color="red", linestyle="--")

    # Add labels for the top 10 lowest score Bad probes
    bad_probes = plot_df[plot_df["Probe category"] == "Bad"].sort_values(col_to_plot).head(10)
    colname = "gene" if "gene" in plot_df.columns else "Probe_ID"

    texts = []
    for index, row in bad_probes.iterrows():
        x = points["x"][points["index"].index(index)]
        y = points["y"][points["index"].index(index)]
        txt = row[colname]
        texts.append(ax.text(x, y, txt, fontsize=8, ha='left', va='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    # Remove x-axis ticks and label
    ax.set_xticks([])
    ax.set_xlabel("")
    #add y label based on col_to_plot
    ax.set_ylabel(col_to_plot)

    # Add a legend if requested
    if legend:
        ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))

    # Convert the plot to HTML
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

    plt.close(fig)
    return html_fig

def not_working_probe_based_on_quality(probe_quality, sample_id="Sample1"):
    probe_quality_neg_probes = probe_quality[probe_quality["Probe_ID"].str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]
    probe_quality_true_probes = probe_quality[~probe_quality["Probe_ID"].str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]
    plot_df = probe_quality.reset_index(drop=True)
    plot_df["Probe category"] = [1 if gene in probe_quality_neg_probes.Probe_ID.values else 0 for gene in plot_df.Probe_ID.values]

    #iterate through the true probes and see whether they are significantly outside of the distribution of the neg probes
    for gene in probe_quality_true_probes.Probe_ID.values:
        plot_df_gene_index = plot_df[plot_df["Probe_ID"]==gene].index[0]
        quality = plot_df[plot_df["Probe_ID"]==gene]["Quality"]
        neg_probes_quality = plot_df[plot_df["Probe category"]==1]["Quality"]
        mean = np.mean(neg_probes_quality)
        std = np.std(neg_probes_quality)
        p_val = stats.norm.cdf(quality, loc=mean, scale=std)
        p_val = 1-p_val
        if p_val*len(probe_quality_true_probes)<0.05:
            plot_df.loc[plot_df_gene_index,"Probe category"] = 2
        
    plot_df["Probe category"] = ["Neg_control" if x==1 else "Bad" if x==0 else "Good" for x in plot_df["Probe category"]]
    plot_df["Sample"] = sample_id
    plot_df = plot_df.sort_values("Probe category")
    return plot_df




def not_working_probe_based_on_morans_i(morans_table, sample_id="Sample1"):
    
    morans_table_neg_probes = morans_table[morans_table.gene.str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]
    morans_table_true_probes = morans_table[~morans_table.gene.str.contains("control|ctrl|pos|neg|Ctrl|blank|Control|Blank|BLANK")]     

    #create a plot_df that is grouped_matrix and a column specifying whether the gene is a neg control or not
    plot_df = morans_table.reset_index(drop=True)
    plot_df["Probe category"] = [1 if gene in morans_table_neg_probes.gene.values else 0 for gene in plot_df.gene.values]

    neg_probes_morans_i_s = plot_df[plot_df["Probe category"]==1]["Morans_I"]
    mean = np.mean(neg_probes_morans_i_s)
    std = np.std(neg_probes_morans_i_s)

    #iterate through the true probes and see whether they are significantly outside of the distribution of the neg probes
    for gene in morans_table_true_probes.gene.values:
        plot_df_gene_index = plot_df[plot_df["gene"]==gene].index[0]
        morans_i_s = plot_df[plot_df["gene"]==gene]["Morans_I"]
        p_val = stats.norm.cdf(morans_i_s, loc=mean, scale=std)
        p_val = 1-p_val
        if p_val*len(morans_table_true_probes)<0.05:
            plot_df.loc[plot_df_gene_index,"Probe category"] = 2
    
    #save the name of those genes with 0
    plot_df["Probe category"] = ["Neg_control" if x==1 else "Good" if x==2 else "Bad" for x in plot_df["Probe category"]]
    plot_df["Sample"] = sample_id
    plot_df = plot_df.sort_values("Probe category")
    return plot_df



def get_unique_features_per_hexagon(matrix_joined):
    unique_features_per_hexagon = matrix_joined.groupby(['x', 'y'])['Gene_ID_y'].nunique().reset_index()
    unique_features_per_hexagon = unique_features_per_hexagon.rename(columns={"Gene_ID_y": "counts"})
    return unique_features_per_hexagon


def get_total_counts_per_hexagon(matrix_joined):
    total_counts_per_hexagon = matrix_joined.groupby(['x', 'y'])['Counts'].sum().reset_index()
    total_counts_per_hexagon = total_counts_per_hexagon.rename(columns={"Counts": "counts"})
    return total_counts_per_hexagon

def get_quality_per_hexagon(matrix_joined):
    hexagon_quality = matrix_joined.groupby("Barcode_ID").agg({"Quality":"first","x":"first","y":"first"})
    hexagon_quality = hexagon_quality.reset_index()
    #rename Quality to counts
    hexagon_quality = hexagon_quality.rename(columns={"Quality": "counts"})
    return hexagon_quality


# Functions used in generate_qc_report
def get_df_for_gene(matrix_joined, tissue_positions_list, gene_name, normalised=False):
    matrix_subset = matrix_joined[matrix_joined["Gene_ID_y"] == gene_name]
    matrix_subset.reset_index(drop=True, inplace=True)
    x = tissue_positions_list["x"]
    y = tissue_positions_list["y"]
    counts = np.zeros(len(tissue_positions_list))
    for i in range(len(matrix_subset)):
        count = matrix_subset["Counts"][i]
        barcode_id = matrix_subset["Barcode_ID"][i]
        x_, y_ = matrix_subset["x"][i], matrix_subset["y"][i]
        x = np.append(x, x_)
        y = np.append(y, y_)

        if normalised:
            count = count / np.sum(matrix_joined[matrix_joined["Barcode_ID"] == barcode_id]["Counts"])

        counts = np.append(counts, count)
    df = pd.DataFrame({"x": x, "y": y, "counts": counts})
    return df.sort_values("counts", ascending=False).drop_duplicates(subset=["x", "y"])



def hexagon_plot_to_html(hexagon_df, hexagon_size, image_pixels_per_um, gene_name, dataset_name,morans_i=None):
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sc = ax.scatter(hexagon_df["x"], hexagon_df["y"], c=hexagon_df["counts"], cmap="viridis", s=2, alpha=0.6)
    for hx, hy in zip(hexagon_df["x"], hexagon_df["y"]):
        #Double check this and why there is the 0.865 division
        hexagon = RegularPolygon((hx, hy), numVertices=6, radius=hexagon_size * image_pixels_per_um / 0.865, alpha=0.2, edgecolor='k', orientation=np.pi / 2)
        ax.add_patch(hexagon)
    ax.set_xlim(0, max(hexagon_df["x"]))
    ax.set_ylim(0, max(hexagon_df["y"]))
    aspect = hexagon_df["x"].max() / hexagon_df["y"].max()
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{gene_name} - {dataset_name}", loc='center', fontsize=8)

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=hexagon_df["counts"].min(), vmax=hexagon_df["counts"].max()))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    if morans_i:
        ax.set_title(f"{gene_name} - Morans I: {morans_i:.2f}", fontsize=8)

    # Convert plot to base64 encoded image string
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

    plt.close(fig)  # Close the plot to free memory

    return html_fig



def get_probe_sums(matrix_joined):
    sums = matrix_joined.groupby("Gene_ID_y").sum()
    sums["Counts"] = np.log10(sums["Counts"])
    sums = sums[["Counts"]]
    sums.reset_index(inplace=True)
    return sums


def plot_sums_to_html(sums1, sums2, dataset1_name, dataset2_name):
    common_probes = list(set(sums1["Gene_ID_y"]) & set(sums2["Gene_ID_y"]))

    if len(common_probes) < 20:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "Not enough overlapping probes", fontsize=12, ha='center')
        ax.axis('off')

        # Convert the plot to HTML
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

        plt.close(fig)  # Close the plot to free memory

        return html_fig
    else:
        # Filter to only contain common probes
        sums1 = sums1[sums1["Gene_ID_y"].isin(common_probes)]
        sums2 = sums2[sums2["Gene_ID_y"].isin(common_probes)]
        # Reorder
        sums1 = sums1.sort_values("Gene_ID_y")
        sums2 = sums2.sort_values("Gene_ID_y")

        sums_df = pd.merge(sums1, sums2, on="Gene_ID_y", suffixes=("_1", "_2"))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(sums_df["Counts_1"], sums_df["Counts_2"], alpha=0.3, s=20)
        ax.set_xlabel(f"Log10 Total expression {dataset1_name}", fontsize=8)
        ax.set_ylabel(f"Log10 Total expression {dataset2_name}", fontsize=8)
        ax.set_title(f"{dataset1_name} vs {dataset2_name}", fontsize=10)

        sums_df["diff"] = np.abs(sums_df["Counts_1"] - sums_df["Counts_2"])
        sums_df = sums_df.sort_values("diff", ascending=False)

        corr = sums_df["Counts_1"].corr(sums_df["Counts_2"])
        corr_text = ax.text(np.quantile(sums_df["Counts_1"], 0.3), np.quantile(sums_df["Counts_2"], 0.95),
                            f"Pearson correlation: {corr:.2f}", fontsize=8)
        texts = [corr_text]
        for i in range(10):
            x = sums_df["Counts_1"].iloc[i]
            y = sums_df["Counts_2"].iloc[i]
            txt = sums_df["Gene_ID_y"].iloc[i]
            texts.append(ax.text(x, y, txt, fontsize=10))

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

        # Convert the plot to HTML
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

        plt.close(fig)  # Close the plot to free memory

        return html_fig



def plot_abundance_correlation_heatmap(replicates_data):
    # Calculate the sums of features for each replicate
    sums_data = []
    replicate_names = []
    for replicate_data in replicates_data:
        sums = get_probe_sums(replicate_data['matrix_joined'])
        sums_data.append(sums)
        replicate_names.append(replicate_data['dataset_name'])
    
    corr_matrix = pd.DataFrame(index=replicate_names, columns=replicate_names, dtype=float)
    min_common_probes = 20
    for i in range(len(sums_data)):
        for j in range(i + 1, len(sums_data)):
            common_probes = list(set(sums_data[i]["Gene_ID_y"]) & set(sums_data[j]["Gene_ID_y"]))
            if len(common_probes) >= min_common_probes:
                # Filter dataframes only for common probes then reorder them to have the same order
                sums1 = sums_data[i][sums_data[i]["Gene_ID_y"].isin(common_probes)]
                sums2 = sums_data[j][sums_data[j]["Gene_ID_y"].isin(common_probes)]
                sums1 = sums1.sort_values("Gene_ID_y")
                sums2 = sums2.sort_values("Gene_ID_y")
                sums_df = pd.merge(sums1, sums2, on="Gene_ID_y", suffixes=("_1", "_2"))
                corr = sums_df["Counts_1"].corr(sums_df["Counts_2"])
                corr = round(corr, 3)
                corr_matrix.iloc[i, j] = corr
                corr_matrix.iloc[j, i] = corr
            else:
                corr_matrix.iloc[i, j] = 0.0
                corr_matrix.iloc[j, i] = 0.0
            
    # Make diagonal 1
    np.fill_diagonal(corr_matrix.values, 1)
    
    fig_dimension = len(replicates_data)
    clustermap = sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', figsize=(fig_dimension, fig_dimension))

    # Convert the plot to HTML
    img_buffer = BytesIO()
    clustermap.savefig(img_buffer, format='png')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

    plt.close(clustermap.figure)
    return html_fig


def get_df_for_gene(matrix_joined, tissue_positions_list, gene_name, normalised=False):
    matrix_subset = matrix_joined[matrix_joined["Gene_ID_y"] == gene_name]
    matrix_subset.reset_index(drop=True, inplace=True)
    x = tissue_positions_list["x"]
    y = tissue_positions_list["y"]
    counts = np.zeros(len(tissue_positions_list))
    for i in range(len(matrix_subset)):
        count = matrix_subset["Counts"][i]
        barcode_id = matrix_subset["Barcode_ID"][i]
        x_, y_ = matrix_subset["x"][i], matrix_subset["y"][i]
        x = np.append(x, x_)
        y = np.append(y, y_)

        if normalised:
            count = count / np.sum(matrix_joined[matrix_joined["Barcode_ID"] == barcode_id]["Counts"])

        counts = np.append(counts, count)
    df = pd.DataFrame({"x": x, "y": y, "counts": counts})
    return df.sort_values("counts", ascending=False).drop_duplicates(subset=["x", "y"])


def process_gene(gene, matrix_joined, tissue_positions_list):
    gene_df = get_df_for_gene(matrix_joined, tissue_positions_list, gene, normalised=True)
    points = [Point(xy) for xy in zip(gene_df['x'], gene_df['y'])]
    gene_gdf = gpd.GeoDataFrame(gene_df, geometry=points)
    w = weights.KNN.from_dataframe(gene_gdf, k=18)
    w.transform = 'R'
    mi = Moran(gene_df["counts"], w, permutations=0)
    return {"gene": gene, "Morans_I": mi.I}

def get_morans_i(gene_name, matrix_joined, tissue_positions_list, max_workers=4):
    if gene_name == "all":
        unique_genes = matrix_joined["Gene_ID_y"].unique()
        #remove those unique genes which have less than total 100 counts
        #lowly_expressed_genes = matrix_joined.groupby("Gene_ID_y")["Counts"].sum()
        #lowly_expressed_genes = lowly_expressed_genes[lowly_expressed_genes<100].index.values
        unique_genes = [gene for gene in unique_genes] #if gene not in lowly_expressed_genes]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_gene, unique_genes, [matrix_joined]*len(unique_genes), [tissue_positions_list]*len(unique_genes)), total=len(unique_genes), desc="Processing genes"))
        
        res_df = pd.DataFrame(results)
        #append the lowly expressed genes to the end of the dataframe with a morans i of 0
        #res_df = pd.concat([res_df,pd.DataFrame({"gene":lowly_expressed_genes,"Morans_I":0})],axis=0)
        
        return res_df
    
    elif gene_name == "features":
        mat = matrix_joined.groupby("Barcode_ID").count()
        mat["x"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["x"].values[0])
        mat["y"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["y"].values[0])
        points = [Point(xy) for xy in zip(mat['x'], mat['y'])]
        gene_gdf = gpd.GeoDataFrame(mat, geometry=points)
        w = weights.KNN.from_dataframe(gene_gdf, k=18)
        w.transform = 'R'
        mi = esda.Moran(mat["Gene_ID_y"], w, permutations=0)
        return mi.I
    
    elif gene_name == "density":
        mat = matrix_joined.copy()
        points = [Point(xy) for xy in zip(mat['x'], mat['y'])]
        gene_gdf = gpd.GeoDataFrame(mat, geometry=points)
        w = weights.KNN.from_dataframe(gene_gdf, k=18)
        w.transform = 'R'
        mi = esda.Moran(mat["counts"], w, permutations=0)
        return mi.I
    
    elif gene_name == "counts":
        mat = matrix_joined.groupby("Barcode_ID").sum()
        mat["x"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["x"].values[0])
        mat["y"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["y"].values[0])
        points = [Point(xy) for xy in zip(mat['x'], mat['y'])]
        gene_gdf = gpd.GeoDataFrame(mat, geometry=points)
        w = weights.KNN.from_dataframe(gene_gdf, k=18)
        w.transform = 'R'
        mi = esda.Moran(mat["Counts"], w, permutations=0)
        return mi.I

    elif gene_name == "Quality":
        mat = matrix_joined.groupby("Barcode_ID").agg({"Quality":"mean"})
        mat["x"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["x"].values[0])
        mat["y"] = mat.index.to_series().apply(lambda barcode: matrix_joined[matrix_joined["Barcode_ID"]==barcode]["y"].values[0])
        points = [Point(xy) for xy in zip(mat['x'], mat['y'])]
        gene_gdf = gpd.GeoDataFrame(mat, geometry=points)
        w = weights.KNN.from_dataframe(gene_gdf, k=18)
        w.transform = 'R'
        mi = esda.Moran(mat["Quality"], w, permutations=0)
        return mi.I
    
    else:
        gene_df = get_df_for_gene(matrix_joined, tissue_positions_list, gene_name, normalised=True)
        points = [Point(xy) for xy in zip(gene_df['x'], gene_df['y'])]
        gene_gdf = gpd.GeoDataFrame(gene_df, geometry=points)
        w = weights.KNN.from_dataframe(gene_gdf, k=18)
        w.transform = 'R'
        mi = esda.Moran(gene_df["counts"], w, permutations=0)
        return mi.I



def plot_morans_i_to_html(morans_i1, morans_i2, dataset1_name, dataset2_name):
    common_probes = list(set(morans_i1["gene"]) & set(morans_i2["gene"]))
    if len(common_probes) < 20:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "Not enough overlapping probes", fontsize=12, ha='center')
        ax.axis('off')

        # Convert the plot to HTML
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

        plt.close(fig)  # Close the plot to free memory

        return html_fig
    else:
        morans_i_df = pd.merge(morans_i1, morans_i2, on="gene", suffixes=("_1", "_2"))
        morans_i_df = morans_i_df[morans_i_df["gene"].isin(common_probes)]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(morans_i_df["Morans_I_1"], morans_i_df["Morans_I_2"], alpha=0.3, s=20)
        ax.set_xlabel(f"Moran's I {dataset1_name}", fontsize=8)
        ax.set_ylabel(f"Moran's I {dataset2_name}", fontsize=8)
        ax.set_title(f"{dataset1_name} vs {dataset2_name}", fontsize=10)

        morans_i_df["diff"] = np.abs(morans_i_df["Morans_I_1"] - morans_i_df["Morans_I_2"])
        morans_i_df = morans_i_df.sort_values("diff", ascending=False)

        corr = morans_i_df["Morans_I_1"].corr(morans_i_df["Morans_I_2"])
        corr_text = ax.text(0.5, 0.1, f"Pearson correlation: {corr:.2f}", fontsize=8, ha='center')

        texts = [corr_text]
        for i in range(10):
            x = morans_i_df["Morans_I_1"].iloc[i]
            y = morans_i_df["Morans_I_2"].iloc[i]
            txt = morans_i_df["gene"].iloc[i]
            texts.append(ax.text(x, y, txt, fontsize=10))

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

        # Convert the plot to HTML
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

        plt.close(fig)  # Close the plot to free memory

        return html_fig


def plot_morans_i_correlation_heatmap(replicates_data):
    #get the morans i for each dataset
    morans_i_data = []
    replicate_names = []
    for replicate_data in replicates_data:
        morans_i = replicate_data["morans_i"]
        morans_i_data.append(morans_i)
        replicate_names.append(replicate_data['dataset_name'])
    
    corr_matrix = pd.DataFrame(index=replicate_names, columns=replicate_names, dtype=float)
    min_common_probes = 20
    for i in range(len(morans_i_data)):
        for j in range(i + 1, len(morans_i_data)):
            common_probes = list(set(morans_i_data[i]["gene"]) & set(morans_i_data[j]["gene"]))
            if len(common_probes) >= min_common_probes:
                #filter dataframes only for common probes then reorder them to have the same order
                morans_i_df = pd.merge(morans_i_data[i], morans_i_data[j], on="gene", suffixes=("_1", "_2"))
                morans_i_df = morans_i_df[morans_i_df["gene"].isin(common_probes)]
                corr = morans_i_df["Morans_I_1"].corr(morans_i_df["Morans_I_2"])
                corr=round(corr,3)
                corr_matrix.iloc[i, j] = corr
                corr_matrix.iloc[j, i] = corr
            else:
                corr_matrix.iloc[i, j] = 0.0
                corr_matrix.iloc[j, i] = 0.0

    #make diagonal 1
    np.fill_diagonal(corr_matrix.values, 1)
    fig_dimension = len(replicates_data) 
    clustermap = sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', figsize=(fig_dimension, fig_dimension))

    # Convert the plot to HTML
    img_buffer = BytesIO()
    clustermap.savefig(img_buffer, format='png')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    html_fig = f'<img src="data:image/png;base64,{img_str}"/>'

    plt.close(clustermap.figure)
    return html_fig



def main():
    parser = argparse.ArgumentParser(description="Generate QC report for Pseudovisium output.")
    parser.add_argument("--folders", "-f", nargs="+", help="List of folders containing Pseudovisium output", required=True)
    parser.add_argument("--output_folder", "-o", default="/Users/k23030440/", help="Output folder path")
    parser.add_argument("--gene_names", "-g", nargs="+", default=["RYR3", "AQP4", "THBS1"], help="List of gene names to plot")
    parser.add_argument("--include_morans_i", "-m", action="store_true", help="Include Moran's I features tab")
    parser.add_argument("-max_workers", "--mw", type=int, default=4, help="Number of workers to use for parallel processing")
    parser.add_argument("-normalisation","--n",action="store_true",help="Normalise the counts by the total counts per cell")

    args = parser.parse_args()

    generate_qc_report(args.folders, args.output_folder, args.gene_names, args.include_morans_i, max_workers=args.mw,normalisation=args.n)


if __name__ == "__main__":
    main()
