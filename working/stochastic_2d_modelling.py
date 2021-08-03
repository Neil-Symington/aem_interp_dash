import numpy as np
import math
import os
import pandas as pd
from garjmcmctdem_utils import spatial_functions
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import imageio


# path to file
infile = "/home/nsymington/Documents/temp/EFTF_seminar_output.csv"

df = pd.read_csv(infile)

# now we want calculate a pdf for each interpreted fiducial

df_baseEvergreen = df[df['OvrStrtUnt'] == "Evergreen"]
df_basePrecipice = df[df['OvrStrtUnt'] == "Precipice"]

# for each unit unit, create a pdf for each fiducial

df_data = pd.DataFrame(columns = ['layer', 'fiducial', 'easting', 'northing', 'DEM', 'mean_elevation', 'stdev'])

for df_ in [df_baseEvergreen, df_basePrecipice]:
    layer = df_['OvrStrtUnt'].unique()
    for fid in df_['fiducial'].unique():
        df_fid = df_[df_['fiducial'] == fid]
        easting = df_fid['X'].unique()
        northing = df_fid['Y'].unique()
        dem = df_fid['DEM'].unique()
        df_fid = df_[df_['fiducial'] == fid]
        p10 = df_fid[df_fid['inversion_name'] == "rj-p10"]['ELEVATION'].values
        p50 = df_fid[df_fid['inversion_name'] == "rj-p50"]['ELEVATION'].values
        p90 = df_fid[df_fid['inversion_name'] == "rj-p90"]['ELEVATION'].values
        # estimate a gaussian
        stdev = np.mean([np.abs(p90 - p50)/2, np.abs(p10 - p50)/2])

        d = {'layer': layer, 'fiducial': fid, 'easting': easting, 'northing': northing, 'mean_elevation': p50,
             'stdev': stdev, 'DEM': dem}
        df_data = df_data.append(pd.DataFrame(d))

# now we will add a synthetic borehole

d = {'layer': ['Precipice'], 'fiducial': ['bore_1'], 'easting': [676019], 'northing': [7151988], 'mean_elevation': [330.0],
             'stdev': [1], 'DEM': [468.9]}

df_data = df_data.append(pd.DataFrame(d))

# sort the data and add a distance along the line
df_data.sort_values(by = 'northing', inplace = True)

# Now get distance along the line

coords = df_data[['easting', 'northing']].values

df_data['dist_along_line'] = spatial_functions.coords2distance(coords)

# Now we want to sample our distributions, generate an ensemble of lines

n = 1000

sampled_points = {}

for surface in ['Precipice', 'Evergreen']:
    sampled_points[surface] = {}
    df_surface = df_data[df_data['layer'] == surface]
    distance = df_surface['dist_along_line'].values
    # Sample the distributions to estimate elevation
    elevs = np.zeros(shape = (len(distance), n), dtype = float)
    mu = df_surface['mean_elevation'].values
    sigma = df_surface['stdev'].values
    # iterate through each point and sample
    for i in range(len(distance)):
        elevs[i] = np.random.normal(mu[i], sigma[i], n)
    sampled_points[surface]['distance'] = distance
    sampled_points[surface]['elevations'] = elevs
    # add the actual input points for plotting
    sampled_points[surface]['mu'] = mu
    sampled_points[surface]['sigma'] = sigma


# we will predict elevation onto 10m distance grids
grid_distances = np.arange(0, math.ceil(df_data['dist_along_line'].max()/10)*10 + 10, 10)

# Now we will create surfaces using a cubic spline

colours = {'Precipice': 'blue', 'Evergreen': 'red'}
outfiles = []

# model ensemble
ensemble = {'Precipice': np.zeros(shape = (n, len(grid_distances)), dtype = float),
           'Evergreen': np.zeros(shape = (n, len(grid_distances)), dtype = float)}

for i in range(n):
    #plt.close('all')
    #fig, ax = plt.subplots(1,1, figsize= (10,5))


    for surface in ['Precipice', 'Evergreen']:

        x = sampled_points[surface]['distance']
        y = sampled_points[surface]['elevations'][:,i]
        cs = CubicSpline(x,y,bc_type = 'clamped')
        grid_elevations = cs(grid_distances)
        #ax.errorbar(x, sampled_points[surface]['mu'], yerr=sampled_points[surface]['sigma'],
        #            c=colours[surface], fmt=" ", barsabove=True)
        #ax.scatter(x, sampled_points[surface]['mu'], c = colours[surface], marker = "o",
        #           s = 5)
        #ax.plot(grid_distances, grid_elevations, label = surface, c = colours[surface])
        #ax.set_ylim(250, 450)
        #ax.set_ylabel('elevation (mAHD)')
        #ax.set_xlabel("distance along line (m)")
        ensemble[surface][i] = grid_elevations


    #ax.legend(loc = 1)
    #outfile = os.path.join("/home/nsymington/Pictures/2d_modelling", "section_{}.png".format(str(i)))
    #plt.savefig(outfile)
    #outfiles.append(outfile)



with imageio.get_writer('/home/nsymington/Pictures/2d_modelling.gif', mode='I') as writer:
    for filename in outfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

# Now we plot a histogram of all stats
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize = (8,8))

ax1.hist(model_stats['area'], bins = 20)
ax1.set_title('area between layers')
ax2.hist(model_stats['min_thickness'], bins = 20)
ax2.set_title('min thickness')
ax3.hist(model_stats['max_thickness'], bins = 20)
ax3.set_title('max thickness')
ax4.hist(model_stats['median_thickness'], bins = 20)
ax4.set_title('median_thickness')

plt.savefig("/home/nsymington/Pictures/stats_histogram.png")

plt.show()



p10 = {}
p50 = {}
p90 = {}

for surface in ['Precipice', 'Evergreen']:
    p10[surface] = np.quantile(ensemble[surface], 0.1, axis=0)
    p50[surface] = np.quantile(ensemble[surface], 0.5, axis=0)
    p90[surface] = np.quantile(ensemble[surface], 0.9, axis=0)

# Plot the quantiles
fig, ax = plt.subplots(1,1, figsize= (10,5))


for i, surface in enumerate(['Precipice', 'Evergreen']):
    if i == 0:
        labels = ['p10', 'p50', 'p90']
    else:
        labels = ['','','',]
    ax.plot(grid_distances, p10[surface], c = 'grey', linestyle = 'dashed', label = labels[0])
    ax.plot(grid_distances, p50[surface],  c = 'k', label = labels[1])
    ax.plot(grid_distances, p90[surface],  c = 'grey', linestyle = 'dashed', label = labels[2])
    ax.errorbar(sampled_points[surface]['distance'], sampled_points[surface]['mu'], yerr=sampled_points[surface]['sigma'],
                c=colours[surface], fmt=" ", barsabove=True)
    ax.scatter(sampled_points[surface]['distance'], sampled_points[surface]['mu'], c = colours[surface], marker = "o",
               s = 5, label = surface)
ax.set_ylim(250, 450)
ax.set_ylabel('elevation (mAHD)')
ax.set_xlabel("distance along line (m)")

ax.legend(loc = 1)
plt.savefig("/home/nsymington/Pictures/quantile_plot.png")
plt.show()




