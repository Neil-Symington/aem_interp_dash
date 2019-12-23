# A script for removing duplicate points from the aseg-gdf directory
# We keep the row with the lowest misift

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

dat = aseg_gdf2.read(infile)

df_AEM = dat.df()

outfile = infile.replace('.dat', '_fixed.dat')

with open(infile, 'r') as inf:
    with open(outfile, 'w') as outf:
        for i, line in enumerate(inf):
            fid = df_AEM.iloc[i]['fiducial ']
            # use mask to find duplicate entries
            df_fid = df_AEM[df_AEM['fiducial '] == fid]
            if len(df_fid) == 1:
                outf.write(line)
            elif df_fid['misfit_lowest '].min() == df_AEM.iloc[i]['misfit_lowest ']:
                outf.write(line)
                print(i)
            else:
                pass

# Now delete the old .dat file and rename the new

os.remove(infile)
os.rename(outfile, infile)
