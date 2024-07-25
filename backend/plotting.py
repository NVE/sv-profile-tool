import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import ipyleaflet
import ipywidgets

from .profile import Profile
from . import soundings
from config import Config, setup_logger

from exceptions import ValueErrorApp


logger = setup_logger(__name__)

cfg = Config()

METHOD_TRANSLATOR = cfg.get("soundings")["method_translator"]
FLUSHING_THRESHOLD = cfg.get("soundings")["flushing_threshold"]
ROTATION_THRESHOLD = cfg.get("soundings")["rotation_threshold"]
MAP_CRS = cfg.get("map")["crs_map"]
COLUMNS = cfg.get("plotting")["columns"]
SOUNDING_PLOT_WIDTH = cfg.get("plotting")["plot_width"]


class ProfileFigure:
    """
    The ProfileFigure class represents a figure of a profile line with boreholes and optional samples.

    Attributes:
        figure (go.FigureWidget): A Plotly FigureWidget containing the profile figure.
        boreholes (gpd.GeoDataFrame): A GeoDataFrame containing the boreholes.
        profile (Profile): A Profile object representing the profile line.
        interp_method (str, optional): The interpolation method to use. Defaults to None.

    Args:
        profile (Profile): The Profile object representing the profile line.
        boreholes (gpd.GeoDataFrame): The GeoDataFrame containing the boreholes.
        samples (gpd.GeoDataFrame, optional): The GeoDataFrame containing the samples. Defaults to None.
        buffer (int, optional): The buffer to use. Defaults to 20.
        layout (go.Layout, optional): The layout to use for the figure. If None, a default layout is created. Defaults to None.
        plot_profile_ratio (int, optional): The ratio of the profile plot's width to its height. Defaults to 2.
        line_depth (int, optional): The depth of the terrain criteria line. Defaults to 0.
        interp_method (str, optional): Which method will be used for interpretation of quick-clay layers. 
                                       If None, no interpretation will be plotted. Defaults to None.

    Raises:
        ValueErrorApp: If the profile line or boreholes are not provided.
    """
    
    def __init__(self, profile:Profile, boreholes:gpd.GeoDataFrame, samples:gpd.GeoDataFrame=None, 
                 buffer:float=20, layout:go.Layout=None, line_depth:float=0, plot_buildings=False,
                 interp_method:str=None, equal_axis_xy:bool=False):

        if layout is None:
            profile_length = round(profile.line.length.iloc[-1], 1)
            graph_width = max(int(profile_length * 4), 500)
            line_coords = profile.line.get_coordinates()
            x0, y0 = line_coords.iloc[0].values
            x1, y1 = line_coords.iloc[1].values
            fig_title = f"Profile @ ({x0:.1f}, {y0:.1f}), ({x1:.1f}, {y1:.1f})"

            layout = go.Layout(
                title=dict(text=fig_title, font=dict(size=12)),
                autosize=False,
                showlegend=True,
                dragmode='pan',
                width=graph_width,
   
            )

        self.figure = go.FigureWidget(layout=layout)
        # self.figure = go.Figure(layout=layout)

        self.boreholes: gpd.GeoDataFrame
        self.profile: Profile

        self.profile = profile
        self.interp_method = interp_method

        if boreholes is None or boreholes.empty:
            self.min_depth = (profile.profile.z.min() * 0.9 // 10) * 10
            self._add_profile_trace()
            self._add_bottom_trace()
            self._add_terraincriteria_line(depth=line_depth, show_local_mins=False)
            
        else:
            self.boreholes = profile.project_points_in_profile(boreholes)
            self.boreholes = self.boreholes.query("dist_profile <= @buffer")

            if samples is not None:
                self.samples = profile.project_points_in_profile(samples)
                # check why i have to dropna here!!
                self.samples = self.samples.dropna(axis=0, subset="method_id").reset_index(drop=True)
                self.samples = self.samples.query("dist_profile <= @buffer")
            else:
                self.samples = None

            min_depth_bh = (self.boreholes.z-self.boreholes.depth).min()
            min_depth_pf = profile.profile.z.min()
            dummy_min_depth = np.nanmin([min_depth_bh, min_depth_pf])

            self.min_depth = (dummy_min_depth * 0.9 // 10) * 10

            self.horizontal_normalization = 30  # 30kN = max in TOT/DTS plots

            self._add_profile_trace()
            self._add_bottom_trace()
            self._add_terraincriteria_line(depth=line_depth, show_local_mins=False)
            self._add_boreholes()
            self._add_samples()

        if plot_buildings:
            self._add_buildings(buffer, use_actual_elevation=True)
        if equal_axis_xy:
            self.figure.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
        


    def show(self) -> None:
        """
        Displays the plot figure.
        """
        plot_config = {'scrollZoom': True}
        self.figure.show(config=plot_config)


    def _add_profile_trace(self) -> None:
        """
        Adds a trace of the terrain profile to the figure.
        """
        self.figure.add_trace(
            go.Scatter(x=self.profile.profile.m, y=self.profile.profile.z, mode="lines",
                       name="terrain surface",
                       line=dict(color="saddlebrown", width=2),
                       showlegend=False)
        )


    def _add_bottom_trace(self) -> None:
        """
        Adds a trace of the terrain bottom to the figure.
        """
        self.figure.add_trace(
            go.Scatter(x=self.profile.profile.m, y=np.ones_like(self.profile.profile.m) * self.min_depth,
                       name="terrain bottom",
                       mode="lines", fill="tonexty", fillcolor="rgba(110, 80, 50, 0.9)",
                       line=dict(color="saddlebrown", width=2),
                       showlegend=False)
        )


    def _add_samples(self) -> None:
        """
        Adds scatter plot traces for the samples to the figure.
        """

        if self.samples is None or self.samples.empty:
            return
        showlegend=True
        for size in [20, 12]:
            self.figure.add_trace(
                go.Scatter(
                    x=self.samples.m_profile,
                    y=self.samples.get_coordinates(include_z=True).z-self.samples.depth,
                    mode="markers",
                    marker=dict(
                        color='rgba(0, 0, 0, 0)',
                        size=size,
                        line=dict(
                            color='rgba(0, 0, 0, 0.5)',
                            width=2
                        )
                    ),
                    # text=[f"{self.samples.location_name}:{self.samples.location_name}"],
                    text=[f"{xx.location_name} @ {xx.depth_top} - {xx.depth_base} m" for xx in self.samples.itertuples()],
                    name="samples",
                    legendgroup="samples",
                    showlegend=showlegend
                )
            )
            showlegend = False


    def _add_terraincriteria_line(self, limit:float=15, depth:float=0, show_local_mins:bool=False) -> None:
        """
        Adds a terrain criteria line to the figure.

        Args:
            depth (int): The depth of the terrain criteria line.
            show_local_mins (bool): Whether to show local minimums on the terrain criteria line.
        """

        (m, z_line), local_mins = self.profile.generate_terraincriteria_line(limit=limit, depth=depth)

        self.figure.add_trace(
            go.Scatter(x=m, y=z_line, line=dict(dash='dash', color="rgba(0, 0, 0, 0.7)", width=0.7, ),
                       mode="lines", 
                       name=f"1:{int(limit)}-line",
                       text=f"1:{int(limit)}-line")
        )
        if show_local_mins:
            self.figure.add_trace(
                go.Scatter(x=local_mins[0], y=local_mins[1],
                           mode="markers", marker=dict(size=3, color="rgba(0, 0, 0, 0.5)"),
                           name="Profile local mins",
                           text="Profile local mins",
                           showlegend=False)
            )


    def _add_boreholes(self) -> None:
        """
        Adds boreholes to the plot.

        This method iterates over the boreholes and adds them to the plot as BoreholePlot objects.
        It also adds interpretations on the profile using the specified geotech. interpretation method.

        """
        for bh in self.boreholes.itertuples():
            try:
                bhplot = BoreholePlot(bh, bh.m_profile, on_profile=True)
            except Exception as e:
                logger.error(f"Error plotting borehole {bh.location_name}: {e}")
                continue
            try:
                bhplot._add_interpretations_on_profile(method=self.interp_method)
            except Exception as e:
                logger.error(f"Error adding interpretations to borehole {bh.location_name}: {e}")
                continue
            self.figure.add_traces(bhplot.figure.data)
                
        try:
            names = [xx.name for xx in self.figure.data]
            index_tolkning = len(names) - names[::-1].index("interpreted quick clay") - 1
            self.figure.data[index_tolkning].update({"showlegend": True})
        except:
            pass
    

    def _add_buildings(self, buffer_distance:float, use_actual_elevation:bool=True) -> None:
        """
        Add buildings (from Geonorge matrikkel bygningspunkt WFS) to the profile plot.
        The markers represent the center of the building projected onto the profile, so some error is expected.

        Args:
            buffer_distance (float): The buffer distance used to select buildings around the profile line.
            use_actual_elevation (bool): Whether to use the actual elevation of the buildings. If False, the elevation is set to the profile z-coordinate.

        Returns:
            None
        """
        from .buildings_api import get_building_points
        from .hoydedata_api import get_z_from_hoydedata

        line = self.profile.line
        bounds = tuple(np.round(line.buffer(buffer_distance).total_bounds))
        buildings = get_building_points(bounds)
        buildings = self.profile.project_points_in_profile(buildings).query("dist_profile<50").copy()
        
        if len(buildings)==0: return

        if use_actual_elevation:
            elevations = get_z_from_hoydedata(buildings.get_coordinates().values)
        else:
            elevations = self.profile.profile.loc[buildings.iloc_profile, "z"]+1

        self.figure.add_trace(
            go.Scatter(
                x=buildings.m_profile, 
                y=elevations,
                marker=dict(
                    symbol="square",
                    color='rgba(0, 0, 0, 0)',
                    size=10,
                    line=dict(color='rgb(0, 0, 0, 0.5)', width=0.5)
                ),
                mode="markers", 
                name="buildings",
                showlegend=True,
                text=[f"dist = {xx:.1f}m" for xx in buildings.dist_profile]
            )
        )



class BoreholePlot:
    """
    A class for creating borehole plots.

    Args:
        borehole (pd.Series): The borehole object containing the data.
        x0 (float, optional): The starting x-coordinate of the plot. Defaults to 0.
        on_profile (bool, optional): Whether to plot the borehole on a profile. Defaults to False.

    Raises:
        ValueErrorApp: If there is no data in the borehole.

    Attributes:
        borehole (pd.Series): The borehole object containing the data.
        x0 (float): The starting x-coordinate of the plot.
        on_profile (bool): Whether the borehole is plotted on a profile.
        plot_width (int): The width of the plot.
        horizontal_normalization (int): The horizontal normalization factor for the load (x) axis.
        z_coordinate (float): The z-coordinate of the borehole.
        figure (go.Figure): The plot figure.

    """
    def __init__(self, borehole, x0=0, on_profile=False):
        
        self.borehole = borehole
        self.x0 = x0
        self.on_profile = on_profile
        
        self.plot_width = 30 if not on_profile else SOUNDING_PLOT_WIDTH
        self.horizontal_normalization = 30

        if borehole.data is None or borehole.data.empty:
            raise ValueErrorApp("No data in this borehole")

        if self.borehole.geometry.has_z:
            self.z_coordinate = self.borehole.geometry.z
        elif hasattr(self.borehole,"z") and not np.isnan(self.borehole.z):
            self.z_coordinate = self.borehole.z
        else:
            self.z_coordinate = 0
        
        fig_title = f"{borehole.method_type.upper()} Borehole {borehole.location_name}"

        layout = go.Layout(
            title=dict(text=fig_title, font=dict(size=12)),
            autosize=True,
            showlegend=True,
            dragmode='pan',
            width=500,
            height=700,
            xaxis=dict(
                tickvals=(np.array([0, 5, 10, 15, 20])*self.plot_width/self.horizontal_normalization + self.x0).tolist(),
                ticktext=['0', '5', '10', '20', '30'],
                range=[-15*self.plot_width/self.horizontal_normalization+self.x0,
                        20*self.plot_width/self.horizontal_normalization+self.x0],
            )
        )

        self.figure = go.Figure(layout=layout)

        self._add_borehole_trace()

    
    def _add_borehole_trace(self) -> None:
        """
        Adds a trace for the borehole data to the figure.

        If the borehole method type is 'cpt', a marker is added to the figure.
        If the borehole method type is not 'cpt', a line plot is added to the figure.

        For 'tot' method type, flagged intervals and extra annotations are added to the figure.

        Returns:
            None
        """
        if self.borehole.method_type == 'cpt':
            if self.on_profile:
                self.figure.add_trace(
                    go.Scatter(
                        x=[self.x0],
                        y=[self.z_coordinate+1],
                        mode="markers",
                        marker=dict(symbol="triangle-down-open", size=20, color="green"),
                        text=[f"{self.borehole.method_type}:{self.borehole.location_name}"],
                        name=f"{self.borehole.method_type}:{self.borehole.location_name}"
                    )
                )
            else:
                # TODO
                return

        else:

            x_plot = self.borehole.data[COLUMNS["force"]].apply(self._scale_force_fn).values / self.horizontal_normalization 
            y_plot = self.z_coordinate - self.borehole.data[COLUMNS["depth"]].values
            m_plot = x_plot * self.plot_width + self.x0


            self.figure.add_trace(
                go.Scatter(x=m_plot, y=y_plot, mode="lines", line=dict(color="black", width=1),
                           text=[f"{self.borehole.method_type}:{self.borehole.location_name}"],
                           name=f"{self.borehole.method_type}:{self.borehole.location_name}",
                           legendgroup=f"{self.borehole.method_type}:{self.borehole.location_name}")
            )

            if self.on_profile:
                self._add_vertical_lines(y_plot)

            if self.borehole.method_type == 'tot':

                self._add_flagged_intervals(kind='ramming')
                self._add_flagged_intervals(kind='flushing')
                self._add_flagged_intervals(kind='rotation')

                if not self.on_profile:
                    self._add_extra_anotations()
                    
            else:
                self.figure.update_xaxes(range=[0 + self.x0, self.plot_width + self.x0])


    def _add_extra_anotations(self) -> None:
        """
        Add extra annotations for hammering, flushing and increase rotation, on the left side of the plot.
        This is valid for totalsoundings only.

        Parameters:
        - None

        Returns:
        - None
        """
        
        for ll,tt in zip([0, -5, -10, -15], [None, "hammer.", "flush.", "inc.rot."]):
            width = 0.5 if ll<0 else 1
            self.figure.add_shape(type="line",
                                        xref="x", yref="paper",
                                        x0=ll, y0=0,
                                        x1=ll, y1=1,
                                        line=dict(color="black",width=width),)
            if ll == 0: continue
            self.figure.add_annotation(x=ll+2.5, y=self.z_coordinate-self.borehole.data[COLUMNS["depth"]].max()-1, 
                                               text=tt, font=dict(size=8, color="black" ), 
                                               align="left", showarrow=False)

    @staticmethod
    def _scale_force_fn(x) -> np.ndarray: 
        """
        Scale force for plotting purposes (equal space between 0-5, 0-10, 10-20 and 20-30)
        """
        return np.where(x > 10, 0.5 * (x - 10) + 10, x)     
           

    def _add_vertical_lines(self, y_plot:float) -> None:
        """
        Add vertical lines to the plot.

        Args:
            y_plot (float): The y-axis values of the plot.
        """

        borehole = self.borehole
        line_0 = self.x0 + \
            np.apply_along_axis(self._scale_force_fn, 0, [0, 0]) / self.horizontal_normalization * self.plot_width
        line_5 = self.x0 + \
            np.apply_along_axis(self._scale_force_fn, 0, [5, 5]) / self.horizontal_normalization * self.plot_width
        line_10 = self.x0 + \
            np.apply_along_axis(self._scale_force_fn, 0, [10, 10]) / self.horizontal_normalization * self.plot_width
        line_20 = self.x0 + \
            np.apply_along_axis(self._scale_force_fn, 0, [20, 20]) / self.horizontal_normalization * self.plot_width
        line_30 = self.x0 + \
            np.apply_along_axis(self._scale_force_fn, 0, [30, 30]) / self.horizontal_normalization * self.plot_width
        lines = [line_5, line_10, line_20, line_30]

        self.figure.add_trace(
            go.Scatter(x=line_0, y=[y_plot.min(), y_plot.max()], mode="lines", line=dict(color="black", width=1),
                       name=f"{borehole.method_type}:{borehole.location_name}",
                       legendgroup=f"{borehole.method_type}:{borehole.location_name}",
                       showlegend=False))

        for li in lines:
            self.figure.add_trace(
                go.Scatter(x=li, y=[y_plot.min(), y_plot.max()], mode="lines", line=dict(color="black", width=0.2),
                           name=f"{borehole.method_type}:{borehole.location_name}",
                           legendgroup=f"{borehole.method_type}:{borehole.location_name}",
                           showlegend=False)
            )


    @staticmethod
    def _get_flagged_intervals(series: pd.Series) -> list:
        """
        Get the flagged intervals from a boolean series.

        Args:
            series (pd.Series): The boolean series.

        Returns:
            list: A list of tuples representing the flagged intervals. Each tuple contains the start and end indices of the interval.
        """
        true_indices = np.where(series)[0]
        groups = np.split(true_indices, np.where(np.diff(true_indices) != 1)[0]+1)
        ranges = [(group[0], group[-1]) for group in groups if len(group) > 0]
        return ranges
    

    def _add_flagged_intervals(self, kind:str='ramming') -> None:
        """
        Adds flagged intervals to the plot based on the specified kind. Only applies total soundings.

        Parameters:
        - kind (str): The type of flagged intervals to add. Must be one of 'ramming', 'flushing', or 'rotation'.

        Returns:
        None
        """
        borehole = self.borehole
        plot_width = self.plot_width

        legendgroup=f"{borehole.method_type}:{borehole.location_name}"
        showlegend=False

        if kind == 'ramming':
            if COLUMNS['ramming_flag'] not in borehole.data.columns:
                print(f'borehole {borehole.location_name} does not have ramming_flag column')
                return
            ranges = self._get_flagged_intervals(borehole.data[COLUMNS['ramming_flag']])
            color = "rgba(1, 90, 56, 0.5)"
            if not self.on_profile:
                x_pos = (0 + -plot_width/6)/2
            else:
                x_pos = round(self.x0-plot_width/25,1)
            x_flag = [x_pos, x_pos]
            name = 'ramming'

        elif kind == 'flushing':
            if COLUMNS["flushing_flag"] not in borehole.data.columns:
                print(f'borehole {borehole.location_name} does not have flush_pressure column')
                return
            ranges = self._get_flagged_intervals(borehole.data[COLUMNS["flushing_flag"]])
            color = "rgba(50, 90, 245, 0.5)"
            if not self.on_profile:
                x_pos = (-plot_width/6+ -plot_width/3)/2
            else:
                x_pos = round(self.x0-plot_width/10,1)
            x_flag = [x_pos, x_pos]
            name = 'flushing'

        elif kind == 'rotation':
            if COLUMNS["rotation_flag"] not in borehole.data.columns:
                print(f'borehole {borehole.location_name} does not have increase rotation column')
                return
            ranges = self._get_flagged_intervals(borehole.data[COLUMNS["rotation_flag"]])
            color = "rgba(255, 0, 204, 0.5)"
            if not self.on_profile:
                x_pos = (-plot_width/3+ -plot_width/2)/2
            else:
                x_pos = round(self.x0-plot_width/6.5,1)
            x_flag = [x_pos, x_pos]
            name = 'rotation'
            
        else:
            raise ValueError("kind must be one of ramming, flushing or rotation")

        if len(ranges) > 0:

            for rr in ranges:
                self.figure.add_trace(
                    go.Scatter(
                        x=x_flag,
                        y=[
                            self.z_coordinate - borehole.data.depth.iloc[rr[0]],
                            self.z_coordinate - borehole.data.depth.iloc[rr[1]],
                        ],
                        mode='lines',
                        line=dict(color=color, width=10 if self.on_profile else 40),
                        name=name,
                        legendgroup=legendgroup,
                        showlegend=showlegend),
                        
                )
                

    def _add_quick_clay_layers(self, quick_clay_layers:pd.DataFrame, x_boxes:float=22.5) -> go.Figure:
        """
        Add quick clay layers from the classify method of one of the intepretation classes to the plot. 

        Parameters:
        - quick_clay_layers (DataFrame): A DataFrame containing information about the quick clay layers.
        - x_boxes (float): The x-coordinate value for the quick clay layers.

        Returns:
        - figure (Figure): The updated figure with the quick clay layers added.
        """

        fill_color = "rgba(255, 0, 0, 0.3)"
        for item in quick_clay_layers.itertuples():
            self.figure.add_trace(
                go.Scatter(
                    x=[x_boxes, x_boxes],
                    y=[
                        self.z_coordinate - item.start,
                        self.z_coordinate - item.stop,
                    ],
                    mode='lines',
                    fillcolor=fill_color,
                    line=dict(color="red", width=40),
                    name="interpreted quick clay",
                    text=["quick clay (int.)"] * 2,
                    legendgroup=f"interpreted quick clay",
                    showlegend=False
                )
            )

        return self.figure


    def _add_quick_clay_interpretation_curves(self, interp_curve:pd.DataFrame, x_boxes:tuple=(0, 20)) -> None:
        """
        Add quick clay interpretation curves (probability or similar) to the plot.

        Parameters:
        - interp_curve (numpy.ndarray): The interpretation curve data.
        - x_boxes (tuple): The x-axis range for plotting the curves.

        Returns:
        None
        """

        x0, x1 = x_boxes
        x_plot = x0 + interp_curve[:, 0] * (x1 - x0)  # curve[0] ranges 0-1
        y_plot = self.z_coordinate - interp_curve[:, 1]

        self.figure.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                line=dict(color="darkgreen", width=1),
                text=np.round(interp_curve[:, 0], 2),
                name="interpretation probability",
                legendgroup=f"interpreted quick clay",
                showlegend=False,
            )
        )


    def plot_interpretations(self, valsson_sens:float=35) -> go.Figure:
        """
        Plot the interpretations for the borehole. Applies only on boreholeplots.

        Args:
            valsson_sens (int): The sensitivity limit for the Valsson classifier.

        Returns:
            plotly.graph_objects.Figure: The plotly figure object with the interpretations plotted.
        """
        if self.on_profile == True:
            raise ValueErrorApp("This method can only be used on a borehole plot. Use _add_interpretations_on_profile instead")
        
        if self.borehole.method_type == 'tot':
            self.figure.update_xaxes(range=[-15,30])
        elif self.borehole.method_type == 'rp':
            self.figure.update_xaxes(range=[0,30])
        else:
            return

        query_borehole = self.borehole

        tc = soundings.ValssonClassifier(query_borehole, sens_limit=valsson_sens)
        qcl_siggy = tc.classify()
        ic_siggy = tc.get_interp_curve()
        self._add_quick_clay_layers(qcl_siggy, x_boxes=27.5)
        self._add_quick_clay_interpretation_curves(ic_siggy, x_boxes=(25,30))

        try:
            gc = soundings.GeoTolkClassifier(query_borehole)
            qcl_gtkl = gc.classify()
            ic_gtkl = gc.get_interp_curve()

            self._add_quick_clay_layers(qcl_gtkl, x_boxes=22.5)
            self._add_quick_clay_interpretation_curves(ic_gtkl, x_boxes=(20,25))
        except:
            pass

        self.figure.add_annotation(x=22.5, y=self.z_coordinate,
            text="GeoTolk", font=dict(size=8, color="black" ), align="left", showarrow=False)
        
        self.figure.add_annotation(x=27.5, y=self.z_coordinate,
            text="Valsson", font=dict(size=8, color="black" ), align="left", showarrow=False)

        self.figure.add_shape(type="line",
                            xref="x", yref="paper",
                            x0=20, y0=0,
                            x1=20, y1=1,
                            line=dict(color="black",width=1,),)
        self.figure.add_shape(type="line",
                            xref="x", yref="paper",
                            x0=25, y0=0,
                            x1=25, y1=1,
                            line=dict(color="black",width=0.5,),)

        try:
            names = [xx.name for xx in self.figure.data]
            index_tolkning = len(names) - names[::-1].index("interpreted quick clay") - 1
            self.figure.data[index_tolkning].update({"showlegend": True})
        except:
            pass
        
        self.figure.update_layout(width=self.figure.layout.width * 1.3)
        return self.figure
    

    def _add_interpretations_on_profile(self, method:str, x_line:float=3, valsson_sens:float=35) -> None:
        """
        Adds interpretations of quick clay layers on the profile plot. Applies only on profile plots.

        Args:
            method (str): The method used for interpreting quick clay layers. Can be "valsson" or "geotolk".
            x_line (int, optional): The x-coordinate where the interpretations will be plotted. Defaults to 3.
            valsson_sens (int, optional): The sensitivity limit for the ValssonClassifier. 
                                           Only applicable if method is "valsson". Defaults to 35.
        """

        if method is None:
            return
        if self.on_profile == False:
            raise ValueErrorApp("This method can only be used on profiles. Use plot_interpretations instead")
        query_borehole = self.borehole

        if query_borehole.method_type not in ("rp", "tot"):
            return

        if method == "valsson":
            classifier = soundings.ValssonClassifier(query_borehole, sens_limit=valsson_sens)
        elif method == "geotolk":
            classifier = soundings.GeoTolkClassifier(query_borehole)
        else:
            return
        quick_clay_layers = classifier.classify()

        x_plot = query_borehole.m_profile + x_line
        for item in quick_clay_layers.itertuples():
            self.figure.add_trace(
                go.Scatter(
                    x=[x_plot, x_plot],
                    y=[
                        query_borehole.z-item.start,
                        query_borehole.z-item.stop,
                    ],
                    mode='lines',
                    line=dict(color="rgba(255, 0, 0, 0.5)", 
                              width=20),
                    name="interpreted quick clay",
                    text=["quick clay (int.)"]*2,
                    legendgroup=f"interpreted quick clay",
                    showlegend=False)
            )


    def _add_samples_to_plot(self, samples_df:pd.DataFrame, x_plot:float=25) -> None:
        """
        Adds samples to the plot.

        Args:
            samples_df (pd.DataFrame): DataFrame containing the lab samples data.
            x_plot (float, optional): The x-coordinate at which the samples will be plotted. Default is 25.

        Returns:
            None
        """
        borehole_loc_id = self.borehole.location_name
        samples = samples_df.query("location_name == @borehole_loc_id").copy()
        color_dict = {"spbr": "rgba(235,158,52,0.5)", "kvkl": "rgba(255,0,0,0.5)", "ikke": "rgba(0,255,0, 0.5)"}
        if "classification" not in samples.columns:
            samples["classification"] = "not_yet"
        sample_types = samples.classification.unique()

        for classification in sample_types:
            sliced_df = samples.query("classification == @classification")
            trace = go.Scatter(
                x=[x_plot] * len(sliced_df),
                y=self.z_coordinate - sliced_df.depth.values,
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=12,
                    color=color_dict.get(classification, "rgba(0,0,0,0)"),
                    line=dict(color="black", width=0.5),
                ),
                text=[classification]* len(sliced_df),
                name="samples",
                legendgroup="samples",
                showlegend=False
            )
            self.figure.add_trace(trace)
        if len(sample_types)>0: self.figure.data[-1].update({"showlegend": True})

    
    def reset_figure(self) -> go.Figure:
        """
        Resets the figure by creating a new instance of `go.Figure` with the same layout and adding the borehole trace.

        Returns:
            go.Figure: The reset figure.
        """
        self.figure = go.Figure(layout=self.figure.layout)
        self._add_borehole_trace()
        return self.figure
    

    def show(self) -> None:
            """
            Displays the plot using the configured plot settings.

            """
            plot_config = {'scrollZoom': True}
            self.figure.show(config=plot_config)


def plotly_to_dxf(fig_data:tuple, filename:str, dxfversion:str='R2018') -> str:
    """
    Converts Plotly figure data to DXF format and saves it as a file.
    Applies to profile plots.

    Args:
        fig_data (list): List of dictionaries containing the figure data.
        filename (str): The name of the output DXF file.
        dxfversion (str, optional): The DXF version to use (see ezdxf documentation). Defaults to 'R2018'.

    Returns:
        str: The filename of the saved DXF file.

    """

    import ezdxf

    dwg = ezdxf.new(dxfversion=dxfversion)
    
    color_dict = {"interpreted quick clay":1, "ramming":62, "flushing":142, "rotation":232}

    x0, y0 = (0, 0)

    skip_layers = ["Profile local mins", "terrain bottom"]

    layers = []
    for trace in fig_data:
        if trace['name'] in skip_layers:
            continue
        name = trace['name'] if trace['name'] is not None else f"other_stuff"
        name = "".join(c for c in name if c.isalpha() or c.isdigit() or c==' ').rstrip()
        name = "_".join(name.split(" "))
        x_coords = np.array(trace['x'])+x0
        y_coords = np.array(trace['y'])+y0
        linestring = list(zip(x_coords, y_coords))
        plot_name_flag = False


        if name in layers:
            dxf_layer = dwg.layers.get(name)
            dxf_layer_name = dxf_layer.dxf.name
        else:
            layers.append(name)
            dwg.layers.new(name=name, dxfattribs={'color': color_dict.get(name, 7)})
            dxf_layer_name = name
            plot_name_flag = True
        
        if trace.mode == "lines":
            dwg.modelspace().add_lwpolyline(linestring, dxfattribs={'layer': dxf_layer_name})
            if trace['name'].startswith(("rp", "tot")) and plot_name_flag:
                dwg.modelspace().add_text(
                    dxf_layer_name, dxfattribs={
                        'layer': dxf_layer_name,  
                        'height': 1.0,  #
                        'insert': (x_coords.min(), y_coords.max()+1),  
                    }
                )

        else:
            for xx,yy in linestring:
                if "cpt" in name:
                    vertices = [(xx, yy), (xx-1,yy+1), (xx+1, yy+1), (xx, yy)]
                    dwg.modelspace().add_lwpolyline(vertices, dxfattribs={'layer': dxf_layer_name})
                    if plot_name_flag:
                        dwg.modelspace().add_text(
                            dxf_layer_name, dxfattribs={
                                'layer': dxf_layer_name,  
                                'height': 1.0,  #
                                'insert': (x_coords.min(), y_coords.max()+2),  
                            }
                        )                    
                else:
                    if name == "samples":
                        dwg.modelspace().add_circle((xx, yy),radius=1,  dxfattribs={'layer': dxf_layer_name})
                        dwg.modelspace().add_circle((xx, yy),radius=0.5,  dxfattribs={'layer': dxf_layer_name})
                    elif name == "buildings":
                        center_x, center_y = xx, yy + 0.5
                        square = [(center_x - 0.5, center_y - 0.5),  
                                  (center_x + 0.5, center_y - 0.5),  
                                  (center_x + 0.5, center_y + 0.5),  
                                  (center_x - 0.5, center_y + 0.5),  
                                  (center_x - 0.5, center_y - 0.5)] 
                        
                        dwg.modelspace().add_lwpolyline(square, dxfattribs={'layer': dxf_layer_name})


    dwg.saveas(filename)
    return filename


def gpd_profiles_to_dxf(gdf:gpd.GeoDataFrame, filename:str, dxfversion:str='R2018') -> str:
    """
    Convert GeoPandas profiles to DXF format and save to a file.

    Parameters:
    - gdf (GeoDataFrame): The GeoPandas DataFrame containing the profiles.
    - filename (str): The name of the output DXF file.
    - dxfversion (str, optional): The DXF version to use. Defaults to 'R2018'.

    Returns:
    - filename (str): The name of the saved DXF file.
    """
    import ezdxf

    dwg = ezdxf.new(dxfversion=dxfversion)
    
    for ii, profile_line in enumerate(gdf.geometry):
        if profile_line is not None:
            dxf_layer_name = f"Profile_{ii+1}_UTM33N"
            dwg.layers.new(name=dxf_layer_name, dxfattribs={'color': 7})
            linestring = list(profile_line.coords)
            dwg.modelspace().add_lwpolyline(linestring, dxfattribs={'layer': dxf_layer_name})
            dwg.modelspace().add_text(
                dxf_layer_name, dxfattribs={
                    'layer': dxf_layer_name,  
                    'height': 2,  #
                    'insert': np.array(list(profile_line.coords)).max(axis=0).tolist(),  
                }
            )  
            dwg.modelspace().add_text(
                'A', dxfattribs={
                    'layer': dxf_layer_name,  
                    'height': 2,  
                    'insert': linestring[0]-np.array([1, 1]), 
                }
            )  

            dwg.modelspace().add_text(
                "A'", dxfattribs={
                    'layer': dxf_layer_name,  
                    'height': 2,  
                    'insert': linestring[-1]+np.array([1, 1]), 
                }
            )  


    dwg.saveas(filename)
    return filename

