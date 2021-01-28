import rasterio
import pickle
import numpy as np
import pandas as pd
import os, glob

# Open up our grids
dir = r"D:\SuratGalilee\2017_SuratGalilee_SkyTEM\final\Delivered_20171024\03_LCI\InJune_MGA55\InJune_MGA55\03_Depth_Slices\Grids_DOI_Masked"
inDir = os.path.join(dir, "*.ers")

# fix dodgy naming
for file in glob.glob(os.path.join(dir, '*')):
    print(file)
    os.rename(file, file.replace('mm', 'm'))


# Dictionary to write results into
layer_grids = {}


nlayers = 30
conversion_to_SI = True

for file in glob.glob(inDir):
    print(file)
    layer = int(file.split('Con')[1].split('_')[0])
    if not layer == nlayers:
        depth_from = float(file.split('Con')[-1].split('_')[-1].split('m.ers')[0].split('-')[0])
        depth_to = float(file.split('Con')[-1].split('_')[-1].split('m.ers')[0].split('-')[1])
    else:
        depth_from = float(file.split('Con')[-1].split('_')[-1].split('m+.ers')[0])

    cond_dataset = rasterio.open(file)
    arr = cond_dataset.read(1)
    arr[arr == cond_dataset.get_nodatavals()] = np.nan
    # convert to S/m
    if conversion_to_SI:
        arr = arr / 1000.
    key = "Layer_" + str(layer)
    layer_grids[key] = {}
    layer_grids[key]['conductivity'] = arr
    layer_grids[key]['depth_from'] = depth_from
    layer_grids[key]['depth_to'] = depth_to

layer_grids['raster_transform'] = cond_dataset.transform

layer_grids['bounds'] = [cond_dataset.bounds.left, cond_dataset.bounds.right,
                        cond_dataset.bounds.bottom,cond_dataset.bounds.top ]

# Add the bounds

outfile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\LCI\grids\Injune_layer_grids.p"

with open(outfile, 'wb') as handle:
    pickle.dump(layer_grids, handle, protocol=2)