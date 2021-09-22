import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from garjmcmctdem_utils import spatial_functions
import matplotlib.patches as patches
from pyproj import Proj, Transformer

dir = r"C:\Users\u77932\Documents\GAB\working\200301"

unit_colours = {'Hutton Sandstone': 'blue', 'Evergreen Formation': 'green',
                    'Precipice Sandstone': 'lightskyblue', 'Rewan Formation': 'pink',
                    'Bandanna Formation': 'orange', 'Boxvale Sandstone Member': 'purple',
                    'Black Alley Shale': 'darkgrey', 'Mantuan Productus Bed': 'lightgrey',
                    'Peawaddy Formation': 'coral', 'Catherine Sandstone': 'pink',
                    'Ingelara Formation': 'red', 'Freitag Formation': 'blanchedalmond',
                     'Aldebaran Sandstone': 'yellow', 'Cattle Creek Formation': 'cyan',
                    'Reids Dome Beds': 'slategray', 'Moolayember Formation': 'salmon',
                    'Clematis Group': 'plum'}

infile = r"E:\GA\dash_data_Surat\CSGwells_coords_TD.csv"

df_bores = pd.read_csv(infile)

# transform the coordinates

transformer = Transformer.from_crs('epsg:4283','epsg:28355', always_xy=True)

lons, lats = df_bores['LONGITUDE'].values, df_bores['LATITUDE'].values

df_bores['easting'], df_bores['northing'] =  transformer.transform(lons, lats)

bore_coords = df_bores[['easting', 'northing']].values

# open the aem data



line = dir.split("\\")[-1]

print(line)

infile = os.path.join(dir, line + ".asc")
aem_coords = np.loadtxt(infile)[None,5:7]

CI = np.log10(1/(10**np.loadtxt(os.path.join(dir,"CI.txt"), delimiter = ',')))

edges = np.log10(1/(10**np.loadtxt(os.path.join(dir,"edges.txt"), delimiter = ',')))[::-1]

himage = np.loadtxt(os.path.join(dir,"himage.txt"), delimiter = ',')[:,::-1]

# normalise

himage = himage/np.sum(himage, axis=1)[0]

depth_tops = np.loadtxt(os.path.join(dir,"depth_tops.txt"))

# since our grid is not rectangular we need to create a coordinate array for mesh depth
zdiffs = np.diff(depth_tops)

mesh_depths = np.hstack((depth_tops, 1.06*zdiffs[-1]/2 + depth_tops[-1])) 


# do a nearest neighbour search
dist, ind = spatial_functions.nearest_neighbours(aem_coords, bore_coords, max_distance = 200)

print(dist)
print(ind)

bore_name = df_bores.iloc[ind[0]]['BORE_NAME']

print(bore_name)

mask = df_bores["BORE_NAME"] == bore_name

df_bore_ss = df_bores[mask]

# now create a figure

fig, ax1 = plt.subplots(1,1, figsize = (5,8))

# now plot the borehole

units = [s[4:] for s in df_bore_ss['STRAT_UNIT'].values]

# ax2

im = ax1.pcolormesh(edges, mesh_depths, himage, cmap = 'Greys')

ax1.plot(CI[:,0], depth_tops, c='grey', linestyle='dashed', linewidth = 0.5, label='p10')
ax1.plot(CI[:,1], depth_tops, c='k', linestyle='solid', linewidth = 0.5, label='p50')
ax1.plot(CI[:,2], depth_tops, c='grey', linestyle='dashed', linewidth = 0.5, label='p90')

for i in range(len(units)):
    break
    # Create a Rectangle patch
    depth_top = df_bore_ss.iloc[i]['MD']
    if depth_top > 250.:
        break
    if i < len(df_bore_ss):
        depth_bottom = df_bore_ss.iloc[i+1]['MD']
        thickness = depth_bottom - depth_top
    else:
        thickness = df_bore_ss['TD'].iloc[i]
    rect = patches.Rectangle((0., depth_top), 0.25, thickness, linewidth=1,
                                 edgecolor='k', facecolor=unit_colours[units[i]],
                                 label = units[i])

    # Add the patch to the Axes
    ax1.add_patch(rect)
ax1.legend(loc = 4)
#480*4
# ax1.set_title('transD quantiles')
ax1.set_ylabel('depth (mBGL)')
ax1.set_xlabel('log10 Conductivity (S/m)')
ax1.grid(which='both')
ax1.set_ylim(250.,0.)
ax1.set_xlim(-3.,1.)

plt.colorbar(im, label = 'probability')

plt.savefig(r"C:\Users\u77932\Documents\GAB\working\plots\{}_AEGC.png".format(bore_name))
