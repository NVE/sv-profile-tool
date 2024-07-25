import requests
import io

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from config import Config, setup_logger

logger = setup_logger(__name__)

cfg = Config()


URL = cfg.get("buildings_api")["url"]
COLS = cfg.get("buildings_api")["columns"]


def get_building_points(bounds):
    gdf = gpd.GeoDataFrame([1], geometry=[box(*bounds)], crs=25833)
    gdf = gdf.to_crs(4326)

    bbox_reprojected = gdf.total_bounds
    xmin, ymin, xmax, ymax = bbox_reprojected 
    
    params = {
        "request": "GetFeature",
        "service": "WFS",
        "version": "2.0.0",
        "typename": "app:Bygning",
        "srsname": "EPSG:25833",
        "bbox": f"{ymin},{xmin},{ymax},{xmax}"
    }
    response = requests.get(URL, params=params)

    if response.status_code == 200:
        file_like_object = io.BytesIO(response.content)
    
    try:
        
        dataset = gpd.read_file(file_like_object)
        dataset = _format_dataset(dataset)
    except Exception as e:
        print(e)
        dataset = gpd.GeoDataFrame(geometry=[], crs=25833)
    
    return dataset


def _format_dataset(dataset):
    out_dataset = dataset.copy()
    out_dataset = out_dataset[COLS].query(cfg.get("buildings_api")["query_filter"])
    return out_dataset