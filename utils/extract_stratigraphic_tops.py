import pandas as pd
import netCDF4
import numpy as np
import geopandas as gpd
import sys, os
sys.path.append("../scripts")
sys.path.append("/home/nsymington/PycharmProjects/garjmcmctdem_utils/scripts")
import aem_utils
import spatial_functions
from shapely.geometry import Point

# Point to AEM inversion netcdf

det = aem_utils.AEM_inversion(name = "lci",
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset("/home/nsymington/Documents/GA/AEM/LCI/Galilee_WB_MGA55.nc"))

infile = "/home/nsymington/Documents/GA/GAB/Galilee/Galilee_strat_tops.csv"

df_strat = pd.read_csv(infile)

df_strat['GA_UNIT'].unique()

long = [x for x in df_strat['Longitude'].values]
lat = [y for y in df_strat['Latitude'].values]

geometry = [Point(long[i],lat[i]) for i in range(len(long))]

gdf_strat = gpd.GeoDataFrame(df_strat, crs="EPSG:4283", geometry=geometry).to_crs("EPSG:28355")

# Remove interfaces of >400m depth

#gdf_shallow = gdf_strat[gdf_strat['TOP_MD_M'] < 400.]

points = np.column_stack(([val.x for val in gdf_strat['geometry']],
                          [val.y for val in gdf_strat['geometry']]))

# Remove data that are not close to the AEM lines
dist, idx = spatial_functions.nearest_neighbours(points, det.coords, points_required = 1,
                                             max_distance = 500.)

# remove invalid indices
real_idx = np.where(np.isfinite(dist))

outfile = "Galilee_strat_proximal_to_AEM.csv"
gdf_strat.iloc[real_idx].to_csv(outfile)

