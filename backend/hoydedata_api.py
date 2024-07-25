import time
import warnings
from urllib.request import urlopen

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio import MemoryFile

from config import Config, setup_logger

logger = setup_logger(__name__)


cfg = Config()

HOYDEDATA_LAYER = cfg.get("profile")["hoydedata_layer"]
HOYDEDATA_URL = cfg.get("profile")["hoydedata_url"]
CRS = cfg.get("map")["crs_default"]  # default crs


def request_hoydedata(bounds, res =5, nodata=-9999, max_retries=5):
    xmin, ymin, xmax, ymax = bounds
    xmin -= 10
    xmax += 10
    ymin -= 10
    ymax += 10

    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)


    request_url = HOYDEDATA_URL.format(HOYDEDATA_LAYER, xmin, ymin, xmax, ymax, width, height, nodata)

    attempts = 0
    wait_time = 1
    while attempts < max_retries:
        try:
            tif_bytes = urlopen(request_url).read()
            break
        except Exception as e:
            attempts += 1
            time.sleep(wait_time)
    else:
        print(request_url)
        raise Exception("Error (Probably area requested is too big/small or høydedata is down)")
    return tif_bytes


def generate_raster_from_hoydedata(tif_bytes):
    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as dataset:
                dem_array = dataset.read(1)
                dataset_profile = dataset.profile

    except rasterio.errors.RasterioIOError as e:
        raise(e)
    
    return dem_array, dataset_profile


def check_hoydedata():
    xmin, ymin, xmax, ymax, width, height, nodata = 261906, 6650936, 264220, 6651626, 462, 138, -9999
    request_url = HOYDEDATA_URL.format(HOYDEDATA_LAYER, xmin, ymin, xmax, ymax, width, height, nodata)


    attempts = 0
    wait_time = 1
    while attempts < 10:
        try:
            _ = urlopen(request_url).read()
            break
        except Exception as e:
            attempts += 1
            time.sleep(wait_time)
    else:
        return False
    return True



def get_z_from_hoydedata(point_array: np.ndarray, res=5) -> np.ndarray:
    """
    Set elevation value to the given x,y points
    Args:
        points_xy: numpy array with the x,y coordinates to the points
        window_data: DEM-results from calling get_hoydedata function

    Returns: numpy array with x,y,z coordinates

    """
    if point_array.shape == (2,):
        points_xy = np.expand_dims(point_array,0)
    else:
        points_xy = point_array.copy()
    xmin, ymin = points_xy.min(axis=0)
    xmax, ymax = points_xy.max(axis=0)

    
    tif_bytes = request_hoydedata((xmin,ymin,xmax,ymax), res=res)
    dem_array, profile = generate_raster_from_hoydedata(tif_bytes)
    transform = profile["transform"]

    index = np.array([rasterio.transform.rowcol(transform, xx[0], xx[1]) for xx in points_xy])
    z = np.array([dem_array[xx[0], xx[1]] for xx in index])

    return z
