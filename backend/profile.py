import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio import MemoryFile
from shapely.geometry import LineString, MultiLineString

from config import Config, setup_logger

from backend.hoydedata_api import request_hoydedata


logger = setup_logger(__name__)

cfg = Config()

HOYDEDATA_LAYER = cfg.get("profile")["hoydedata_layer"]
HOYDEDATA_URL = cfg.get("profile")["hoydedata_url"]
CRS = cfg.get("map")["crs_default"]  # default crs


class Profile:
    """
    The Profile class represents a topographic profile from a line, using Høydedata as source.

    Attributes:
        line (gpd.GeoDataFrame): A GeoDataFrame containing the line in the specified CRS.

    Args:
        line (gpd.GeoDataFrame, LineString, MultiLineString, or np.ndarray): The line to be represented. 
        This can be a GeoDataFrame, a LineString, a MultiLineString, or a numpy array of shape (n, 2).
        crs (str, optional): The CRS to use. Defaults to CRS.

    Raises:
        ValueError: If line is a numpy array but its shape is not (n, 2).
    """
    
    def __init__(self, line, crs=CRS):
        """
        Initializes the Profile object with a line and a CRS.

        Args:
            line (gpd.GeoDataFrame, LineString, MultiLineString, or np.ndarray): The line to be represented. This can be a GeoDataFrame, a LineString, a MultiLineString, or a numpy array of shape (n, 2).
            crs (str, optional): The CRS to use. Defaults to CRS.

        Raises:
            ValueError: If line is a numpy array but its shape is not (n, 2).
        """

        if isinstance(line, gpd.GeoDataFrame):
            line_gdf = line.to_crs(CRS)

        elif isinstance(line, LineString) or isinstance(line, MultiLineString):
            line_gdf = gpd.GeoDataFrame(geometry=[line], crs=crs).to_crs(CRS)

        elif isinstance(line, np.ndarray):
            if line.shape[1] != 2:
                raise ValueError("if line is array shape should be (n,2)")
            line_gdf = gpd.GeoDataFrame(geometry=[LineString(line)], crs=crs).to_crs(CRS)

        else:
            print(type(line))
            raise ValueError()

        self.line = line_gdf

        self._interpolate(dist=5, min_points=5)
        self.points_coords = self.line.get_coordinates().values

        self.profile = self._profile()


    def _interpolate(self, dist=5, min_points=5):
        """
        Interpolates points along the line geometry.

        Args:
            dist (float): The distance between interpolated points.
            min_points (int): The minimum number of points to interpolate.

        Returns:
            None

        Raises:
            None
        """

        line_geometry = self.line.iloc[0].geometry
        n_points = int(max(line_geometry.length // dist, min_points))

        new_points = [line_geometry.interpolate(i / float(n_points - 1), normalized=True) for i in range(n_points)]
        points_coords = np.array([[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points])

        self.line['geometry'] = LineString(points_coords)


    def _get_hoydedata(self):
        """
        Retrieves elevation data for the given points.

        Returns:
            list: A list of elevation values corresponding to the given points.
        """

        tif_bytes = request_hoydedata(tuple(self.line.total_bounds))

        z_dem = []
        try:
            with MemoryFile(tif_bytes) as memfile:
                with memfile.open() as dataset:
                    dem_array = dataset.read(1)
                    for pp in self.points_coords:
                        ind = dataset.index(pp[0], pp[1])
                        z_dem.append(dem_array[ind[0], ind[1]])

        except rasterio.errors.RasterioIOError as e:
            print("feil: ", e)

        return z_dem


    def _profile(self):
        """
        Calculates the profile of the points based on elevation data.

        Returns:
            pandas.DataFrame: A DataFrame containing the x, y, z, and m values of the profile.
                - x: x-coordinates of the points
                - y: y-coordinates of the points
                - z: elevation data of the points
                - m: cumulative distance along the profile
        """

        z_dem = self._get_hoydedata()

        cum_dist = np.cumsum(
            np.sqrt(np.sum((np.r_[[[0, 0]], np.diff(self.points_coords, axis=0)[:, :2]]) ** 2, axis=1)))

        self.profile = pd.DataFrame({'x': self.points_coords[:, 0],
                                     'y': self.points_coords[:, 1],
                                     'z': z_dem,
                                     'm': cum_dist})

        return self.profile


    def project_points_in_profile(self, points_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Projects a point geodataframe onto a profile line, using the shortest distance.

        Args:
            points_df (GeoDataFrame): The DataFrame containing the points to be projected.

        Returns:
            GeoDataFrame: A tuple containing the projected points DataFrame.

        """
        out_gdf = points_df.copy()
        out_gdf = out_gdf.to_crs(CRS)

        profile_xy = self.points_coords

        bh_coordinates = out_gdf.get_coordinates(include_z=True)

        def distance(xy): return np.sqrt((bh_coordinates.x.values -
                                          xy[0]) ** 2 + (bh_coordinates.y.values - xy[1]) ** 2)

        dists = np.array([distance(p_xy) for p_xy in profile_xy])

        min_dist = np.min(dists, axis=0)  # minimum distance
        i_pt_min = np.argmin(dists, axis=0)  # index of point at minimum distance
        point_data_profile = self.profile.loc[i_pt_min, :]  # profile data of point at minimum distance

        point_data_profile['dist_profile'] = min_dist
        point_data_profile['iloc_profile'] = point_data_profile.index
        point_data_profile['m_profile'] = point_data_profile.m

        out_gdf = (out_gdf.reset_index(drop=True)
                          .join(point_data_profile[['iloc_profile', 'm_profile', 'dist_profile']].reset_index(drop=True),
                                                      how='right'))

        return out_gdf


    def generate_terraincriteria_line(self, limit=15, depth=0, res=5)->tuple:
        """
        Generate a line with the terrain criteria (by default 1:15). The line is computed from all the interpolated 
        points of the profile, with interpolation resolution given by res.

        Args:
            limit (float): Limit in vertical/horizontal ratio (1:limit) to be used for the terrain criteria.
            depth (float): Depth to be used for the line.
            res (float): Resolution of the line.

        Returns:
            two tuples with m,z values for both the line and the local minima of the profile
        """
        # from scipy.signal import argrelextrema
        z = self.profile.z
        m = self.profile.m
        length = m.max()-m.min() #self.line.length.sum()

        num_points = int(max(length//res, 10))
        
        m_interp = np.linspace(m.min(), m.max(), num_points)
        z_interp = np.interp(m_interp, m, z)
        
        # # Old code from when we used local minima instead of every point
        # local_mins = list(argrelextrema(z_interp, np.less, order=order)[0])
        # local_mins.append(np.argmin(z_interp)) # ensure one value

        # local_mins = tuple(local_mins)

        # lines = []

        # local_mins_list_m = [m_interp[ll] for ll in local_mins]
        # local_mins_list_z = [z_interp[ll] for ll in local_mins]

        lines = []
        local_mins = range(len(m_interp))
        local_mins_list_m = m_interp
        local_mins_list_z = z_interp

        for i_min in local_mins:
            z_min = z_interp[i_min]
            m_0 = m_interp[i_min]
            m_line = np.abs(m_interp - m_0)
            z_line = m_line / limit
            lines.append(z_min + z_line - depth)

        z_line_out = np.min(lines, axis=0)
        z_line_out[z_line_out > z_interp] = z_interp[z_line_out > z_interp]

        return (m_interp, z_line_out), (local_mins_list_m, local_mins_list_z)


    def plot(self, ax=None):
        """
        Plots the profile data on the given axes.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the profile data. If not provided, 
            a new set of axes will be created.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plotted profile data.
        """
        if ax is None:
            ax = self.profile.plot(x='m', y='z')
        else:
            self.profile.plot(x='m', y='z', ax=ax)
        ax.axis('equal')
        return ax

