# A script for mapping points from the rjmcmc.dat file to their pmap matlab files

import geopandas as gpd
import numpy as np
import pandas as pd
import aseg_gdf2
from shapely.geometry import Point
import sys
import glob
import os

#load file as pandas dataframe using .dfn file

infile = sys.argv[1]
pmap_dir = sys.argv[2]

dat = aseg_gdf2.read(infile)

df_AEM = dat.df()

xs = df_AEM['easting '].values
ys = df_AEM['northing '].values

df_AEM['geometry'] = [Point(xs[i],ys[i]) for i in range(len(xs))]

gdf_AEM = gpd.GeoDataFrame(df_AEM)

# Keep only 1D columns

cols = [x for x in df_AEM.columns if not '[' in x]

gdf_AEM = gdf_AEM[cols]

# Now we want to get a link to the pmap files

gdf_AEM['matfile'] = ''

# iterate through the geodataframe and find the files
# based on the naming convention

pmap_files = {}

for filename in glob.glob(os.path.join(pmap_dir, "*.mat")):
    # Get the fid
    fid = np.float('.'.join([filename.split('.')[3],
                            filename.split('.')[4]]))
    pmap_files[fid] = filename

print(pmap_files)
# Iterate through the geodataframe
for index, row in gdf_AEM.iterrows():
    fid = np.float(row['fiducial '])
    # Use this fid to find the pmap file from the directory
    for item in pmap_files.keys():
        if np.round(item,1) == np.round(fid,1):
            gdf_AEM.at[index, 'matfile'] = pmap_files[item]
            continue
    
        
# Export the file

gdf_AEM.to_csv(infile + '_map.csv', index = False)