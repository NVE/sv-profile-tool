import copy
import time
import requests
import asyncio
import httpx

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import Config, setup_logger

from exceptions import StopExecution


logger = setup_logger(__name__)
cfg = Config()

METHOD_DICT = cfg.get("fm_api")["method_dict"]
METHOD_DICT_REV = cfg.get("fm_api")["method_dict_rev"]
METHOD_STATUS = cfg.get("fm_api")["method_status"]
METHOD_TYPES_FILTER = eval(cfg.get("fm_api")["method_types_filter"]) # so it will be a tuple and not a str

ENCODING = cfg.get("fm_api")["encoding"]
REQUEST_LIMIT = cfg.get("fm_api")["request_limit"]


class FMApiClient:
    def __init__(self, token):
        self.base_url = 'https://api.fieldmanager.io/fieldmanager/'
        self._token = token
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        if not self._check_api_access():
            raise StopExecution()

        self.method_types = self._get_method_types()

        _ = self.get_organizations()
        _ = self.get_projects()

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, value):
        self._token = value
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        self._check_api_access()

    def _get_method_types(self):
        r = self._get(self.base_url+"method_types")
        return r.json()

    def _check_api_access(self):

        response = requests.get(self.base_url, headers=self.headers)

        if response.status_code == 401:
            print("\033[91m\033[1m")
            print("\nToken invalid or expired! Get a new one at: ")
            print("\033[0m")
            print("https://app.fieldmanager.io/developer")
            return False
        else:
            return True

    def _get(self, url, params={'limit': REQUEST_LIMIT}):
        if not self._check_api_access():
            return

        response = requests.get(url, headers=self.headers, params=params)
        return response

    async def _get_async(self, url_list, params={'limit': REQUEST_LIMIT}):
        timeout = httpx.Timeout(120)
        async with httpx.AsyncClient(timeout=timeout) as client:
            rr = [client.get(url, params=params, headers=self.headers) for url in url_list]
            results = await asyncio.gather(*rr)
        return results


    def get_organizations(self):
        if not self._check_api_access():
            return

        url = self.base_url + 'organizations'
        response = self._get(url)

        r_json = response.json()
        org_df = pd.DataFrame.from_dict(r_json)

        self.organizations = org_df
        return org_df
    

    def get_projects(self):
        if not self._check_api_access():
            return

        url = self.base_url + f'projects'
        response = self._get(url)

        r_json = response.json()

        prj_df = pd.DataFrame.from_dict(r_json)

        self.projects = prj_df
        return prj_df
    

    def get_locations(self, project_id=None, name=None, iloc=None):
        if not self._check_api_access():
            return
        if 'projects' not in self.__dict__.keys():
            self.get_projects()

        if project_id is not None:
            selected_project = self.projects.query("project_id == @project_id")
            prj_id = project_id
        elif name is not None:
            selected_project = self.projects.query("name == @name")
            prj_id = selected_project.project_id
        elif iloc is not None:
            selected_project = self.projects.iloc[iloc]
            print(f"retrieving project '{selected_project['name']}'")
            print(f"project_id '{selected_project.project_id}'")
            prj_id = selected_project.project_id

        crs = selected_project.srid

        url = self.base_url + f'projects/{prj_id}/locations'

        response = self._get(url)
        r_json = response.json()
        locs_df = pd.DataFrame.from_dict(r_json)
        geometry = gpd.points_from_xy(x=locs_df.point_easting, y=locs_df.point_northing, z=locs_df.point_z, crs=crs)
        locs_gdf = gpd.GeoDataFrame(geometry=geometry,
                                    data=locs_df.drop(
                                        columns=['point_easting', 'point_northing',
                                                 'point_x_wgs84_pseudo', 'point_y_wgs84_pseudo',
                                                 'point_x_wgs84_web', 'point_y_wgs84_web']))
        locs_gdf["methods_n"] = locs_gdf.methods.apply(len)
        locs_gdf["methods_name"] = locs_gdf.methods.apply(
            lambda x: [METHOD_DICT_REV.get(
                xx['method_type_id']
            ) for xx in x if len(x) > 0] if len(x) > 0 else [])

        self.locations = locs_gdf
        return locs_gdf
    

    async def get_location_methods(self):

        locs_gdf = self.locations.copy()
        prj_id = list(locs_gdf.project_id.values)
        loc_id = list(locs_gdf.location_id.values)
        url_list = [self.base_url+f'projects/{pp}/locations/{ll}/methods' for pp, ll in zip(prj_id, loc_id)]
        results = await self._get_async(url_list)
        results = [xx for xx in map(lambda x: x.json(), results)]

        methods_list = []
        for methods_dict in results:
            sub_method_list = []
            if len(methods_dict) > 0:
                for item_dict in methods_dict:
                    method = METHOD_DICT_REV.get(item_dict['method_type_id'], 'NA')
                    if method not in sub_method_list:
                        sub_method_list.append(method)
            methods_list.append(sub_method_list)

        return methods_list
    

    async def get_dataset(self):
        if self.locations is None:
            raise ValueError("no locations fetched, run get_locations first")
        
        dict_list = []

        for loc in self.locations.itertuples():
            methods_i = loc.methods
            if len(methods_i)>0:
                dict_list.extend(methods_i)
        methods = pd.DataFrame(dict_list)
        methods["method_type"] = methods.method_type_id.map(METHOD_DICT_REV)
        methods = methods.query("method_type in @METHOD_TYPES_FILTER").copy()

        url = self.base_url + "projects/{}/locations/{}/methods/{}/data"

        prj_id = self.locations.project_id.unique()[0]
        url_list = [url.format(prj_id, ll, mm) for ll, mm in methods[["location_id", "method_id"]].values]
        print("retrieving soundings and samples, please wait...")
        start_time = time.perf_counter()
        data = await self._get_async(url_list)
        end_time = time.perf_counter()
        print(f"data retrieved in {(end_time-start_time):.1f} seconds")

        
        methods["data"] = [pd.DataFrame(rr.json()) if rr.status_code==200 else None for rr in data]

        loc_dict = self.locations[["location_id", "geometry"]].copy().set_index("location_id").to_dict(orient="index")
        loc_names_dict = self.locations[["location_id", "name"]].copy().set_index("location_id").to_dict()["name"]

        geometry = methods.location_id.map(loc_dict)
        geometry = geometry.apply(lambda x: x["geometry"])

        methods["location_name"] = methods.loc[:, "location_id"].map(loc_names_dict)


        methods["geometry"] = geometry

        return gpd.GeoDataFrame(methods, crs=self.locations.crs)


    def plot_locations(self):

        map = self.locations.explore(
            tooltip=['name', 'methods_n', 'methods_name'],
            popup=['name', 'created_at', 'created_by', 'updated_at', 'updated_by',
                   'tags', 'is_deleted', 'last_updated', 'methods_n', 'methods_name'],
            marker_kwds={'radius': 5})

        return map


    @staticmethod
    def _format_snd(saved_file):
        with open(saved_file, "r") as file:
            lines = file.readlines()
        lines = [line for line in lines if line.strip()]
        with open(saved_file, 'w') as file:
            file.writelines(lines)

    @staticmethod
    def _fix_snd_parsed_data(data_list):
        data_list_copy = copy.deepcopy(data_list)
        for data in data_list_copy:
            df = data.get("data", [])
            if len(df) == 0:
                continue
            fixed_df = df.copy()
            fixed_df["feed_thrust_force"] = fixed_df["feed_thrust_force"]/1000
            try:
                fixed_df["rotation_speed"] = fixed_df["extra_spin"]*100
                fixed_df["ramming_flag"] = fixed_df["ramming"]
                fixed_df["flush_pressure"] = fixed_df["flushing"]*100
            except KeyError:
                pass

            data["data"] = fixed_df
        return data_list_copy
