import glob
import os
import tempfile
import warnings
import zipfile

import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm_notebook as tqdm
from shapely.geometry import Point


from config import Config, setup_logger
import exceptions

logger = setup_logger(__name__)


cfg = Config()
fgdb_params = cfg.get("nadag_fgdb")

BASIC_LAYERS = set(["GeotekniskUnders", "GeotekniskBorehull", "GeotekniskBorehullUnders", "StatiskSondering",
                   "KombinasjonSondering", "Trykksondering", "StatiskSonderingData", "KombinasjonSonderingData",
                   "TrykksonderingData", "GeotekniskProveserie", "GeotekniskProveseriedelData", 
                   "GeotekniskProveseriedel"])


def open_nadag_fgdb(fgdb_path:str, verbose=True) -> gpd.GeoDataFrame:
    layers = fiona.listlayers(fgdb_path)

    GeotekniskUnders = gpd.read_file(fgdb_path, layer="GeotekniskUnders", engine="pyogrio")
    GeotekniskBorehull = gpd.read_file(fgdb_path, layer="GeotekniskBorehull", engine="pyogrio")
    GeotekniskBorehullUnders = gpd.read_file(fgdb_path, layer="GeotekniskBorehullUnders", engine="pyogrio")

    StatiskSondering = gpd.read_file(fgdb_path, layer="StatiskSondering", engine="pyogrio") if "StatiskSondering" in layers else None
    KombinasjonSondering = gpd.read_file(fgdb_path, layer="KombinasjonSondering", engine="pyogrio") if "StatiskSondering" in layers else None
    Trykksondering = gpd.read_file(fgdb_path, layer="Trykksondering", engine="pyogrio") if "StatiskSondering" in layers else None

    StatiskSonderingData = gpd.read_file(fgdb_path, layer="StatiskSonderingData", engine="pyogrio") if "StatiskSonderingData" in layers else None
    KombinasjonSonderingData = gpd.read_file(fgdb_path, layer="KombinasjonSonderingData", engine="pyogrio") if "StatiskSonderingData" in layers else None
    TrykksonderingData = gpd.read_file(fgdb_path, layer="TrykksonderingData", engine="pyogrio") if "StatiskSonderingData" in layers else None

    # TODO:
    # GeotekniskProveseriedelData = gpd.read_file(fgdb_path, layer="GeotekniskProveseriedelData", engine="pyogrio") if "GeotekniskProveseriedelData" in layers else None
    # GeotekniskProveseriedel = gpd.read_file(fgdb_path, layer="GeotekniskProveseriedel", engine="pyogrio") if "GeotekniskProveseriedel" in layers else None
    # GeotekniskProveserie = gpd.read_file(fgdb_path, layer="GeotekniskProveserie", engine="pyogrio") if "GeotekniskProveserie" in layers else None

    crs = GeotekniskBorehull.crs

    sonderingdata_fk = ["statisksondering_fk", "kombinasjonsondering_fk", "trykksondering_fk"]
    sonderingstype = ["StatiskSondering", "KombinasjonSondering", "Trykksondering"]

    df_list = []
    for item in tqdm(GeotekniskBorehull.itertuples(), total=len(GeotekniskBorehull), desc="Reading NADAG-FGDB", disable=not verbose):
        bhid = item.lokalid
        bhgusid = item.opprinneliggeotekniskundersid
        gusid = GeotekniskUnders.query("lokalid == @bhgusid").lokalid.iloc[0]
        try:
            for bhuid in GeotekniskBorehullUnders.query("geotekniskborehull_fk == @bhid").lokalid.values:
                data_to_field = [None]
                for ss, sondering in enumerate([StatiskSondering, KombinasjonSondering, Trykksondering]):
                    
                    if sondering is None: 
                        continue

                    sondering_data = [StatiskSonderingData, KombinasjonSonderingData, TrykksonderingData][ss]

                    ksid_query = sondering.query("geotekniskborehullunders_fk == @bhuid").lokalid

                    if len(ksid_query) == 0:
                        data_to_field = [None]
                    else:

                        for ksid in ksid_query.values:

                            data = sondering_data[sondering_data[sonderingdata_fk[ss]] == ksid].copy()
                            if len(data) == 0:
                                continue
                            else:
                                if ss == 3:
                                    data["alpha"] = sondering.query("lokalid == @ksid").alpha.iloc[0]
                                fk_unique = data[sonderingdata_fk[ss]].unique()
                                data_to_field = [data.loc[sondering_data[sonderingdata_fk[ss]] == fk] for fk in fk_unique]
                                data_to_field = [dd.replace({np.nan:None}) for dd in data_to_field]


                            df_0 = pd.DataFrame({"lokal_id": item.lokalid, "geotekniskunders_id":gusid, "sonderingstype": sonderingstype[ss], "data": data_to_field})
                            df_list.append(df_0)
        except IndexError:
            print("IndexError: ", f"{bhid=}", f"{bhgusid=}")
            continue

    df = pd.concat(df_list).dropna(subset=["data"], axis=0)
    df["height"] = df.lokal_id.apply(lambda x: GeotekniskBorehull.query("lokalid == @x").iloc[0].hoyde)
    df["kvkl"] = df.lokal_id.apply(lambda x: GeotekniskBorehull.query("lokalid == @x").iloc[0].kvikkleirepavisning)
    df["geometry"] = df.lokal_id.apply(lambda x: GeotekniskBorehull.query("lokalid == @x").iloc[0].geometry)
    df["borenr"] = df.lokal_id.apply(lambda x: GeotekniskBorehull.query("lokalid == @x").iloc[0].borenr)

    df["oppdragstaker"] = df.geotekniskunders_id.apply(lambda x: GeotekniskUnders.query("lokalid == @x").iloc[0].oppdragstaker)
    df["prosjektnavn"] = df.geotekniskunders_id.apply(lambda x: GeotekniskUnders.query("lokalid == @x").iloc[0].prosjektnavn)

    df = df.reset_index(drop=True)
    df = gpd.GeoDataFrame(df)
    df.crs = crs

    return df


def create_flagged_column(df, col_start, col_end):
    changes = pd.Series(0, index=df.index)
    changes[df[col_start]] = 1
    changes[df[col_end]] = -1
    
    cum_state = changes.cumsum()
    
    return cum_state > 0


def create_intervals_from_comments(input_df):
    df = input_df.copy()
    flag_codes = fgdb_params["flag_codes"]


    df["comment_code"] = df["comment_code"].map(lambda x: str(int(x)) if x is not None and not isinstance(x, str) else x)
    for col, codes in flag_codes.items():
        df[col] = df["comment_code"].apply(lambda x: any(code in x for code in codes) if x is not None else False)

    for col in ["hammering", "increased_rotation_rate", "flushing"]:
           df[col] = create_flagged_column(df, col+"_starts", col+"_ends")
    
    return df[["hammering", "increased_rotation_rate", "flushing"]]


def setup_soundings(fgdb:gpd.GeoDataFrame, verbose=True) -> tuple:

    
    method_type_mapper = fgdb_params["sonderingstype_to_method_type"]
    data_columns_mapper = fgdb_params["data_to_data"]

    method_type = fgdb["sonderingstype"].map(method_type_mapper)
    geometry = [Point(x, y, z) for x, y, z in zip(fgdb.geometry.x, fgdb.geometry.y, fgdb['height'])]
    location_name = fgdb["borenr"]
    data_list = fgdb.data.values.tolist()
    data_list_as_fm = []
    for ii, elem in tqdm(enumerate(data_list), total=len(data_list), desc="Converting FGDB file", disable=not verbose):
        new_elem = (
            elem.rename(columns=data_columns_mapper)
                .sort_values(by="depth")
                .reset_index(drop=True)
                    )
        if method_type[ii] == 'tot':
            new_elem[["hammering", "increased_rotation_rate", "flushing"]] = create_intervals_from_comments(new_elem)
        else:
            new_elem[["hammering", "increased_rotation_rate", "flushing"]] = False

        data_list_as_fm.append(new_elem)

    out_dict = {
        "method_type": method_type,
        "geometry": geometry,
        "location_name": location_name,
        "data": data_list_as_fm,

    }

    boreholes = gpd.GeoDataFrame(out_dict, crs=fgdb.crs)
    boreholes[["x", "y", "z"]] = boreholes.get_coordinates(include_z=True)
    boreholes["depth"] = boreholes.data.apply(lambda x: x["depth"].max())
    boreholes["method_id"] = boreholes.data.apply(lambda x: x["method_id"].iloc[0])
    boreholes["method_status"] = "conducted"
    boreholes["method_status_id"] = 3

    #TODO:
    #Is CPT compatible? Check if Units are the same!


    #TODO
    samples = None


    return boreholes, samples
    

def read_zipped_fgdb(zip_path:str) -> tuple:
    """
    Reads a zipped File Geodatabase (FGDB) and returns the formated datasets for boreholes and samples.

    Args:
        zip_path (str): The path to the zipped FGDB file.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: A tuple containing the boreholes and samples.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            try:
                gdb_file = glob.glob(tmpdir+"/*.gdb")[0]
            except IndexError:
                raise exceptions.ValueErrorApp("Error in zipped file, make sure that you have zipped just the .gdb file (not the folder containing it)")
            print("Processing GDB Dataset: ", os.path.basename(gdb_file).split(".")[0])
            ds = open_nadag_fgdb(gdb_file)
            bh, ss = setup_soundings(ds)

    return bh, ss