import pandas as pd
import netCDF4
import numpy as np
import geopandas as gpd
import sys, os
from garjmcmctdem_utils import spatial_functions, aem_utils
from shapely.geometry import Point

# Point to AEM inversion netcdf

det = aem_utils.AEM_inversion(name = "lci",
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset("/home/nsymington/Documents/GA/AEM/LCI/Injune_lci_MGA55.nc"))

infile = "/home/nsymington/Documents/GA/dash_data_Surat/CSGwells_coords_TD.csv"

df_strat = pd.read_csv(infile)

long = [x for x in df_strat['LONGITUDE'].values]
lat = [y for y in df_strat['LATITUDE'].values]

geometry = [Point(long[i],lat[i]) for i in range(len(long))]

gdf_strat = gpd.GeoDataFrame(df_strat, crs="EPSG:4283", geometry=geometry).to_crs("EPSG:28355")

gdf_strat['easting'] = gdf_strat['geometry'].x
gdf_strat['northing'] = gdf_strat['geometry'].y

# Remove interfaces of >400m depth

gdf_strat = gdf_strat[gdf_strat['MD'] < 400.]

points = np.column_stack(([val.x for val in gdf_strat['geometry']],
                          [val.y for val in gdf_strat['geometry']]))

# Remove data that are not close to the AEM lines
dist, idx = spatial_functions.nearest_neighbours(points, det.coords, points_required = 1,
                                             max_distance = 100.)

# remove wells too far from the lines
bore_idx = np.where(np.isfinite(dist))

# Find the line number and fiducial of the nearest points
aem_idx = idx[bore_idx]

# get the AEM elevation
elev = det.data['elevation'][aem_idx]

fids = det.data['fiducial'][aem_idx].data

lines = det.data['line'][det.data['line_index'][aem_idx]]

gdf_strat_ss = gdf_strat.iloc[bore_idx]
gdf_strat_ss['fiducial'] = fids
gdf_strat_ss['line'] = lines
gdf_strat_ss['elevation'] = elev

# do some final nipping and tucking

gdf_strat_ss['Strat_name'] = [s[4:] for s in gdf_strat_ss['STRAT_UNIT']]
gdf_strat_ss['GA_UNIT'] = gdf_strat_ss['Strat_name'].copy()

gdf_strat_ss['TOP_AHD_M'] = elev - gdf_strat_ss["MD"].values

outfile = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_strat_proximal_to_AEM.csv"

gdf_strat_ss.to_csv(outfile, index= False)
