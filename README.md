# sv-profile-tool
# NVE Profile Tool for Nadag and Field Manager

This repository is for sharing the backend of the [profile-tool](https://profile-tool.azurewebsites.net). 

The profile tool is a web app for visualizing and interpreting geotechnical data. The app is developed for NVE, and is used for visualizing and interpreting geotechnical data (total and pressure soundings) from NADAG or Field Manager, and is especially useful for quick-clay hazard assessments.

The profiles shows the terrain model, soundings (total, rotary pressures, and where cpts and samples are), and the lowest 1:15-line from a given depth. This last part is especially useful for quick-clay hazard assessments. Profiles can be exported and downloaded as dxf.

Tha app also includes the possibility to interpret the soundings and identify where it might be quick or sensitive clay. The interpretation uses the Valsson et al (2004) chart method for total/pressure soundings. The method is not published yet, and it is only an indication of where quick clay might be, and won't replace a proper geotechnical assessment.

The backend uses Pandas/Geopandas, Rasterio, Numpy, and Shapely for the calculations.
