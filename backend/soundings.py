import warnings

from itertools import groupby

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from tqdm.notebook import tqdm


from . import hoydedata_api
from config import Config, setup_logger


logger = setup_logger(__name__)

cfg = Config()

CRS = cfg.get("map")["crs_default"]
SAMPLES_TYPES = tuple(cfg.get("fm_api")["samples_types"])
SOUNDINGS_TYPES = tuple(cfg.get("fm_api")["sounding_types"])
METHOD_STATUS = cfg.get("fm_api")["method_status"]

METHOD_TRANSLATOR = cfg.get("soundings")["method_translator"]

COLS = cfg.get("plotting")["columns"]
THICKNESS_LIMIT = cfg.get("soundings")["thickness_limit"]
MIN_LENGTH_DATA = cfg.get("soundings")["min_length_sounding_data"]


def setup_soundings(methods: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Preprocesses boreholes and samples data fetched from the FieldManager API (fm_api module) to be used in the frontend.

    Args:
        methods (gpd.GeoDataFrame): A GeoDataFrame containing methods data from fm_api module.

    Returns:
        tuple: A tuple containing two GeoDataFrames - boreholes and samples.

    """

    # preprocessing of boreholes:
    boreholes = (
                methods.query("method_type in @SOUNDINGS_TYPES")
                .copy()
                .to_crs(CRS)
                .dropna(subset=["data"])
    )
    boreholes = boreholes[boreholes["data"].apply(lambda x: not x.empty)]
    boreholes["depth"] = _get_borehole_depth(boreholes)
    boreholes[['x', 'y', 'z']] = boreholes.get_coordinates(include_z=True).round(1)

    boreholes_no_z = boreholes[np.isnan(boreholes.z)][["x","y"]].values
    if len(boreholes_no_z)>0:
        z_from_hd = hoydedata_api.get_z_from_hoydedata(boreholes_no_z, res=5)
        boreholes.loc[np.isnan(boreholes.z), "z"] = z_from_hd

    boreholes["method_status"] = boreholes.method_status_id.map(METHOD_STATUS)
    boreholes = boreholes.query("method_status in ('approved', 'conducted')").copy()
    boreholes = boreholes[boreholes.data.apply(len)>MIN_LENGTH_DATA].reset_index(drop=True)


    # preprocessing of samples:
    samples = (
                methods.query("method_type in @SAMPLES_TYPES")
                .copy()
                .to_crs(CRS)
    )
    samples["depth"] = _get_sample_depth(samples)
    samples[['x', 'y', 'z']] = samples.get_coordinates(include_z=True).round(1)

    samples_no_z = samples[np.isnan(samples.z)][["x","y"]].values
    if len(samples_no_z)>0:
        z_from_hd_samples = hoydedata_api.get_z_from_hoydedata(samples_no_z, res=5)
        samples.loc[np.isnan(samples.z), "z"] = z_from_hd_samples

    samples["method_status"] = samples.method_status_id.map(METHOD_STATUS)
    samples = (samples.query("method_status in ('approved', 'conducted')")
                      .copy()
                      .reset_index(drop=True))


    return boreholes, samples


def _get_borehole_depth(gdf: gpd.GeoDataFrame) -> list:
    """
    Get the maximum depth of each borehole in the GeoDataFrame.

    Parameters:
    gdf (gpd.GeoDataFrame): A GeoDataFrame containing borehole data.

    Returns:
    list: A list of maximum depths for each borehole. If a borehole has no data, None is appended.

    """
    depth = []
    for dd in gdf.itertuples():
        if not (dd.data is None or dd.data.empty):
            depth.append(dd.data.depth.max())
        else:
            depth.append(None)
    return depth


def _get_sample_depth(gdf: gpd.GeoDataFrame)->list:
    """
    Calculate the average depth for samples with method_type 'sa' in the given GeoDataFrame.

    Parameters:
    gdf (gpd.GeoDataFrame): The GeoDataFrame containing the samples.

    Returns:
    list: A list of average depths for the samples.
    """

    depth = ((gdf.query("method_type == 'sa'").depth_top + gdf.query("method_type == 'sa'").depth_base)/2).values
    return list(depth)


def _correct_rp_bool_columns(borehole:gpd.GeoDataFrame):
    """
    Check the RotaryPressure sounding data for the given borehole and update the flags to False if necessary.
    this is because some RP soundings come with increase rotation = True, and that makes the interpretation inaccurate.

    Args:
        borehole (GeoDataFrame): The Borehole object containing the data to be checked.

    Returns:
        None
    """
    data = borehole.data
    if borehole.method_type != "rp":
        return
    columns = [xx for xx in data.columns if xx in (COLS["ramming_flag"], COLS["flushing_flag"], COLS["rotation_flag"])]
    for col in columns:
        if data[col].any():
            #print("updated to False ", col)
            data[col] = False
    

def _add_missed_columns_sounding(borehole_data:pd.DataFrame):
    """
    Check the Total and RotaryPressure soundings data and ensures that they have all the boolean columns 
    (ramming, flushin and rotation) so the data can be used for interpretation.

    Args:
        borehole_data (DataFrame): The Borehole data object containing the data to be checked.

    Returns:
        None
    """
    data = borehole_data.copy()

    for col in [COLS["ramming_flag"], COLS["flushing_flag"], COLS["rotation_flag"]]:
        if col not in data.columns:
            data[col] = False

    return data


class ValssonClassifier():
    """
    Wrapper class for handling the interpretation using Valsson's method.

    Attributes:


    Args:
        fm_borehole (pd.Series): The borehole element from fm_api boreholes dataframe, after using setup_soundings function.
                                     It is just one element of the dataframe.
        sens_limit (float): limit for the sensitivity/p-index of the interpretation. Lower limit makes interpretation 
                            more conservative.
        delta_h (float): layer thickness for averaging sounding values.

    """
    
    def __init__(self, fm_borehole:gpd.GeoDataFrame, sens_limit:float=35, delta_h:float=1):
        
        self.borehole = fm_borehole
        data = self.borehole.data

        if fm_borehole.method_type == "rp":
            _correct_rp_bool_columns(self.borehole)

        self.data = (data.copy()
                        .dropna(axis=1, how="all")
                        .sort_values(by=COLS["depth"])
                        .reset_index(drop=True))
        if COLS["comments"] in self.data.columns:
            self.data = self.data.drop(columns=COLS["comments"])
        if not all(x in self.data.columns for x in [COLS["force"], COLS["depth"]]):
            raise ValueError("depth and force columns must be present in the data columns.")
        
        self.data = _add_missed_columns_sounding(self.data)
                    
        self.sens_limit = sens_limit
        self.delta_h = delta_h

    
    @staticmethod
    def _get_n_per(df: pd.DataFrame, dh: float, min_periods: int = 10) -> int:
        """
        Calculate the number of periods based on the thickness of the layers.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            dh (float): The value to divide by the maximum difference in depth.
            min_periods (int, optional): The minimum number of periods. Defaults to 10.

        Returns:
            int: The calculated number of periods.
        """

        dbl = df.sort_values(by=COLS["depth"])[COLS["depth"]].diff().max()

        try:
            n_per = max(min_periods, int(dh / dbl)) if not np.isnan(dbl) else min_periods
        except OverflowError:
            n_per = min_periods
        return n_per
    

    def _get_classification_data(self) -> tuple:
        """
        Calculate classification data (rolling averaged) based on the given parameters.

        Returns:
            z (numpy.ndarray): Array of depth values.
            fdt (numpy.ndarray): Array of mean force values.
            qn (numpy.ndarray): Array of normalized cone resistance values.
            std_fdt (numpy.ndarray): Array of standard deviation of force values.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            df = self.data.copy()
            area = 2.55e-3 #m2 drilling bit
            gamma = 7 #kN/m3 soil density
            n_per = self._get_n_per(df, self.delta_h, 10)

            dd = df[[COLS["force"] ,COLS["depth"]]].sort_values(by=COLS["depth"]).copy()

            # This could be done to correct the force values, but after talking to Valsson, we decided to not do it.
            # The reason is that sometimes there are soundings that show negative values at the end, and correcting 
            # for that would make the interpretation more conservative (and probably wrong).

            # force_correction_factor = min(dd[COLS["force"]].min(), 0)
            # dd[COLS["force"]] = dd[COLS["force"]] - force_correction_factor

            z = dd.rolling(n_per, min_periods=0).mean()[COLS["depth"]].values
            fdt = dd.rolling(n_per, min_periods=0).mean()[COLS["force"]].values
            std_fdt = dd.rolling(n_per, min_periods=0).std()[COLS["force"]].values

            qn = fdt/(area * gamma * z)


        return z, fdt, qn, std_fdt


    def predict(self, qns: np.ndarray, std_fdt: np.ndarray) -> np.ndarray:
        """
        Predicts the value of 'p' based on the given 'qns' and 'std_fdt' values, using Valsson's chart method.
        This function is taken from Valsson's code (SimpleSensClassifier).

        Args:
            qns (np.ndarray): A list of qns values.
            std_fdt (np.ndarray): A list of std_fdt values.

        Returns:
            np.ndarray: An array of predicted 'p' values.

        Raises:
            ZeroDivisionError: If 'std_fdt' contains any zero values.

        """
        a = 2/0.0089
        b = -0.61435
        y0 = 0.0006
        x0 = 40

        p_res = [0] * len(qns)
        for i, (some_qns, some_std_fdt) in enumerate(zip(qns, std_fdt)):
            # p as a variable for better readability
            p = a * (np.arctan(np.log10(some_std_fdt/y0) / (np.sqrt((np.log10(some_qns/x0))**2 + (np.log10(some_std_fdt/y0))**2) + np.log10(some_qns/x0))) + b)
            p_res[i] = p
            
        return np.array(p_res)


    def classify(self) -> pd.DataFrame:
            """
            Classify the sounding data into layers based on specified criteria.
            The result is (relatively) standardized so it can be used for plotting.

            Returns:
                layers (pd.DataFrame): List of layers identified in the sounding data.
            """

            sonderings_data = self.data.copy()
            if sonderings_data is None:
                return np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) 


                depth_rock = sonderings_data.query("comment_code==94")[COLS["depth"]].min() if COLS["comments"] in sonderings_data.columns else 9999
                depth_rock = min( depth_rock if not np.isnan(depth_rock) else 9999, sonderings_data[COLS["depth"]].max()+1)

                z, _, qn, std_fdt = self._get_classification_data()

                p = self.predict(qn, std_fdt)

                
                # mask = sonderings_data[[COLS["flushing_flag"], COLS["ramming_flag"], COLS["rotation_flag"]]].any(axis=1).values.tolist()
                # feedback from Valsson is that increased rotation does not affect the interpretation
                mask = sonderings_data[[COLS["flushing_flag"], COLS["ramming_flag"]]].any(axis=1).values.tolist() 
                mask = self._interpolate_flags(sonderings_data[COLS["depth"]], z, mask)
                
                p[mask] = 0
                layers = self._find_layers(p, self.sens_limit, z)
                self.p_index = p
                self.z = z

                return layers
    
    @staticmethod
    def _interpolate_flags(z_original: np.ndarray, z_new: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """
        Interpolates the (boolean) flags to the new depth values.

        Args:
            z_original (np.ndarray): The original depth values.
            z_new (np.ndarray): The new depth values.
            flags (np.ndarray): The flags to be interpolated.

        Returns:
            np.ndarray: The interpolated flags.

        """

        flag_interp = np.interp(z_new, z_original, np.array(flags).astype(int))

        return flag_interp.astype(bool)


    def get_interp_curve(self) -> np.ndarray:
        """
        Returns the interpretation curve for defining the layers. 
        This can be used to calibrate the results of the interpretation.

        Returns:
            numpy.ndarray: The interpolated curve, represented as a 2D array with shape (n, 2),
            where n is the number of elements in p_index and z attributes.
            The first column represents the curve_0 values, and the second column represents the z values.
        """
        curve_0 = self.p_index/100
        curve_0[curve_0<0] = 0
        interp_curve = np.c_[curve_0, self.z]

        return interp_curve
    

    @staticmethod
    def _find_layers(p_index:np.ndarray, sens:float, depth:np.ndarray) -> pd.DataFrame:
        """
        Find continuous layers in the given interpretation data based on a sensitivity/probability threshold.

        Args:
            p_index (np.ndarray): An array of p-index values.
            sens (float): The sensitivity threshold for the p-index.
            depth (np.ndarray): An array of depth values.

        Returns:
            pd.DataFrame: A DataFrame containing information about the layers found, including start and stop depths,
                          layer thicknesses, and mean pressure index values.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pred = p_index > sens
            groups = groupby(enumerate(pred), key=lambda x: x[1])
            continuous_ones = [list(group) for key, group in groups if key == 1]
            indices = [(group[0][0], group[-1][0]) for group in continuous_ones]

            starts = [depth[layer[0]] for layer in indices]
            ends = [depth[layer[1]] for layer in indices]
            thicknesses = [end - start for start, end in zip(starts, ends)]
            p_index_mean = [np.mean(p_index[layer[0]:layer[1]]) for layer in indices]

            df = pd.DataFrame({"start": starts,
                               "stop": ends,
                               "thickness": thicknesses,
                               "p_mean": p_index_mean})
            df = df.dropna(axis=0)
        return df
       

class GeoTolkClassifier:
    """
    A class for classifying borehole data using the GeoTolk API.
    
    Args:
    fm_borehole (pd.Series): The borehole element from fm_api boreholes dataframe, after using setup_soundings function.
                                 It is just one element of the dataframe.

    
    Attributes:
        borehole (gpd.GeoSeries): The borehole element for the boreholes dataframe from fm_api (after using sounding.setup_soundings function).
        df (DataFrame): The formatted DataFrame used for querying the GeoTolk api.
        layers (list): The quick clay layers identified by the classification.
        static (dict): The static data, including probability, returned by the GeoTolk API.
    
    Raises:
        ValueError: If the borehole does not have a dataset.
    """
    
    def __init__(self, fm_borehole):
        from . import geotolk_api
        self.borehole = fm_borehole
        query_df_data = self.borehole.data

        if fm_borehole.method_type == "rp":
            # we found some rp soundings that had 'increased_rotation_pressure', so we correct this before continuing
            _correct_rp_bool_columns(self.borehole)

        if query_df_data.empty:
            raise ValueError("Borehole does not have a dataset")
        
        self.df = geotolk_api.format_df_to_api(query_df_data)
    
    def classify(self):
        """
        Classifies the borehole data using the GeoTolk API.
        
        Returns:
            list: The quick clay layers identified by the classification.
        """
        from . import geotolk_api
        response_dict = geotolk_api.query_geotolk(self.df)

        self.layers = response_dict["QUICK_CLAY_LAYERS"]
        self.static = response_dict["STATIC_DATA"]

        return self.layers
    
    def get_interp_curve(self):
        """
        Retrieves the interpolated curve from the static data.
        
        Returns:
            numpy.ndarray: The interpolated curve with depth and Udrenert values.
        """
        interp_curve = self.static[["Udrenert", "Dybde"]].values
        interp_curve[:,0] = interp_curve[:,0]/100

        return interp_curve


def interpret_boreholes_valsson(boreholes_query, limit_quick=35, limit_not_quick=50):
    quick_clay = []
    max_thickness = np.zeros((len(boreholes_query)))
    error_list = []
    
    for ii,(_, row) in tqdm(enumerate(boreholes_query.iterrows()), total=len(boreholes_query), desc="Interpreting Boreholes"):
        if len(row.data)<5 or row.data.penetration_force.isnull().all():
            quick_clay.append("error")
            error_list.append(f"Too few data or null force at {ii}, location {row.location_name}")
            continue
        try:
            vc = ValssonClassifier(row, sens_limit=limit_quick)
            layers = vc.classify()
        except Exception as e:
            error_list.append(f"Unknown error at {ii}, location {row.location_name}")
            quick_clay.append("error")
            continue
        if len(layers)==0 or layers.thickness.max()<1:
            quick_clay.append("not quick clay")
            continue
        vc = ValssonClassifier(row, sens_limit=limit_not_quick)
        layers = vc.classify()
        if layers.thickness.max()>0.5:
            quick_clay.append("quick clay")
            max_thickness[ii] = layers.thickness.max()
        else:
            quick_clay.append("unsure")
    if len(error_list)>0:
        logger.info(f"finished with {len(error_list)} errors:")
        for line in error_list:
            logger.info(line)
    return quick_clay, max_thickness


