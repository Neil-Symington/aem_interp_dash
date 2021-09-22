import numpy as np
import pandas as pd
import os
from garjmcmctdem_utils import spatial_functions
from pyproj import Proj, Transformer

# open bores

#infile = r"E:\GA\dash_data_Surat\CSGwells_coords_TD.csv"

#df_bores = pd.read_csv(infile)


#bore_names = ["FAIRVIEW 77", "SPRING ROCK 1", "FAIRVIEW 533", "FAIRVIEW 123 OB1"]

#for item in bore_names:
#    if not item in df_bores['BORE_NAME'].values:
#        print("{} not found".format(item))

#df_bores_ss = df_bores[np.isin(df_bores['BORE_NAME'].values, bore_names)]

# transform the coordinates

#transformer = Transformer.from_crs('epsg:4283','epsg:28355', always_xy=True)

#lons, lats = df_bores['LONGITUDE'].values, df_bores['LATITUDE'].values

#df_bores['easting'], df_bores['northing'] =  transformer.transform(lons, lats)

# get the coordiantes of the inversion
infile = r"C:\Users\u77932\Documents\MORPH\data\AEM\2016_SkyTEM\working\rho_mid_line_103601_summary_xyzrho.txt"

coords = np.unique(np.loadtxt(infile, usecols = [0,1]),axis=0)

# bring in the AEM data

aem_infile = r"C:\Users\u77932\Documents\MORPH\data\AEM\2016_musgraves_tempest\located_data\Musgrave_Region_Final_EM.dat"

aem_coords = np.loadtxt(aem_infile, usecols = (10,11))

# convert to zone 53

transformer = Transformer.from_crs('epsg:28352','epsg:28353', always_xy=True)

X, Y = coords[:,0], coords[:,1]

new_X, new_Y =  transformer.transform(X,Y)

new_coords = np.column_stack((new_X, new_Y))

# run a nearest neighbour to find the nearest AEM fiducial

dist, ind = spatial_functions.nearest_neighbours(new_coords, aem_coords, max_distance = 20.)

assert np.isfinite(dist)

# now we export a new file with only the rows specified in ind

outfile = r"C:\Users\u77932\Documents\MORPH\data\AEM\2016_musgraves_tempest\working\Musgrave_tempest_testline_EM.dat"

with open(aem_infile, 'r') as f:
    with open(outfile, 'w') as outf:
        for i, line in enumerate(f):
            if i in ind:
                outf.write(line)

