import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from garjmcmctdem_utils import spatial_functions, aem_utils, netcdf_utils, misc_utils, modelling_utils
from garjmcmctdem_utils import plotting_functions as plots
import netCDF4
from shapely.geometry import MultiPoint, Point
from osgeo import gdal
import gc


from scipy.interpolate import CloughTocher2DInterpolator

def idw(x,y,v, X,Y, power = 2, max_distance = 500.):
    coords = np.column_stack((x,y))
    grid_coords = np.column_stack((X,Y))
    # find the distances
    dist, ind = spatial_functions.nearest_neighbours(grid_coords, coords,
                                                     points_required = 6, max_distance=max_distance)
    # make the nulls from dist equal zero
    ind[np.isnan(dist)] = 0
    # get inverse distance and find the weights
    id = 1/(dist**power)
    id[np.isnan(id)] = 0
    weights = id/np.sum(id,axis = 1)[:,None]
    # multiply the normalised distance by the values
    values = (weights * v[ind]).sum(axis = 1)
    return values

infile = r"E:\GA\GAB\Injune\quantile_interp_consolidated.csv"

df_interp = pd.read_csv(infile)

df_interp.dropna(how = 'any', subset = ['p10','p90'], inplace=True)

df_interp['p50_mAHD'] = df_interp['DEM'] - df_interp['p50']

df_interp['depth_stdev'] = np.abs(df_interp['p90'] - df_interp['p10'])/2.5

df_interp["depth_runc"] = df_interp['depth_stdev']/df_interp['p50']

df_interp[]

nc_infile = r"E:\GA\dash_data_Surat\Injune_pmaps_reduced_concatenated.nc"

rj = aem_utils.AEM_inversion(name = 'rj', inversion_type = 'stochastic', netcdf_dataset = netCDF4.Dataset(nc_infile))

fiducials = df_interp['fiducial'].values

df_interp['point_index'] = -999
#df_interp['p50_conductance'] = np.nan
#df_interp['p10_conductance'] = np.nan
#df_interp['p90_conductance'] = np.nan
df_interp['cond_stdev'] = np.nan
df_interp['min_phid'] = np.nan

for index, row in df_interp.iterrows():
    fid = row['fiducial']
    df_interp.at[index, 'point_index'] = np.argmin(np.abs(fid - rj.data['fiducial'][:]))

# # calculate the conductance at each point
# for index, row in df_interp.iterrows():
#     point_index = row['point_index']
#     cond_profile = 10**rj.data['conductivity_p50'][point_index]
#     min_phid = rj.data['misfit_lowest'][point_index]

#     depth = row['p50']
#     depth_mask = np.where(rj.data['layer_top_depth'][:] < depth)[0]
#     conductance = np.sum((cond_profile[depth_mask] * rj.data['layer_top_depth'][depth_mask]))/np.sum(rj.data['layer_top_depth'][depth_mask])
#     df_interp.at[index, 'p50_conductance'] = conductance
#     df_interp.at[index, 'cond_stdev'] = np.std(cond_profile[depth_mask])
#     df_interp.at[index, 'min_phid'] = min_phid



# fig, ax = plt.subplots(1,1,figsize = (8,8))
# sc = ax.scatter(df_interp['p50'].values,df_interp['depth_stdev'].values,
#                 c = df_interp['p50_conductance'].values)
# ax.set_xlabel('depth uncertainty (m)')
# ax.set_ylabel('layer interface depth (m)')
# plt.colorbar(sc, label = 'conductance')

# # now   run a linear regression
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(df_interp['p50'].values,
#                                                                df_interp['depth_stdev'].values,)
# print(r_value**2)
# x = np.array([0,300])
# y = slope*x + intercept
# ax.plot(x, y, label = "linear regression function")
# plt.show()

# fig, ax = plt.subplots(1,1, figsize = (8,8))
# ax.hist(df_interp['depth_runc'].values, bins = 30)
# ax.set_ylabel('freq')
# ax.set_xlabel('relative depth uncertainty')
# plt.show()

# df_interp['depth_stdev_2'] = slope*df_interp['p50'].values + intercept

# fig, ax = plt.subplots(1,1, figsize = (8,8))
# # find how muh the results skew
# skewness = (df_interp['p50'] - df_interp['p10']) - (df_interp['p90'] - df_interp['p50']).values

# ax.hist(skewness, bins = 20)
# ax.set_ylabel('freq')
# ax.set_xlabel('skewness (m)')
# plt.show()


surfaces = {"TopPrecipice": {},
 "BasePrecipice":{}}

# first get the coords from the Petrel grid
for surface_name in ["TopPrecipice", "BasePrecipice"]:
    src = gdal.Open(r"E:\GA\GAB\interpretation\{}.tif".format(surface_name))

    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    xcoords = np.linspace(ulx, lrx, src.RasterXSize)
    ycoords = np.linspace(lry, uly, src.RasterYSize)
    gridX, gridY = np.meshgrid(xcoords, ycoords)
    arr = np.array(src.GetRasterBand(1).ReadAsArray())
    surfaces[surface_name]['petrel_values'] = arr
    surfaces[surface_name]['petrel_coords'] = np.column_stack((gridX.flatten(),gridY.flatten()))


coords = df_interp[['X', 'Y']].values   

# These will be interpolated onto a common grid
X = np.arange(np.min(coords[:,0]) - 100., np.max(coords[:,0]) + 100., 100.)
Y = np.arange(np.min(coords[:,1]) - 100., np.max(coords[:,1]) + 100., 100.)

# Create the grid for each surface


X, Y = np.meshgrid(X,Y)

grid_coords = np.column_stack((X.flatten(), Y.flatten()))

n = 1000


for surface_name in ["TopPrecipice", "BasePrecipice"]:

    mask = df_interp['surface'] == surface_name
    # create a convex hull
    surface_coords = df_interp[mask][['X','Y']].values
    points = [Point(pt) for pt in surface_coords]
    mpt = MultiPoint(points)
    surfaces[surface_name]['convex_hull'] = mpt.convex_hull.buffer(100.)
    n = 1000
    mu_elev = df_interp[mask]['DEM'].values - df_interp[mask]['p50'].values
    stdev = df_interp[mask]['depth_stdev'].values

    surfaces[surface_name]['values'] = np.nan * np.ones(shape = (n, X.shape[0], X.shape[1]), dtype = float)

    for i in range(n):

        values = idw(surface_coords[:,0], surface_coords[:,1], np.random.normal(mu_elev, stdev),
                                                  grid_coords[:,0],grid_coords[:,1],
                                                  power = 2, max_distance = 10000.)
        surfaces[surface_name]['values'][i] = values.reshape(X.shape)
    # finally mask values not within the convex hull
    mask = np.zeros(shape = len(grid_coords), dtype = bool)
    for i in range(len(grid_coords)):
        [x,y]= grid_coords[i]
        if Point(x,y).within(surfaces[surface_name]['convex_hull']):
            mask[i] = True
    surfaces[surface_name]['values'][:,~mask.reshape(X.shape)] = np.nan

    # plot the mean
    fig, ax = plt.subplots(1,1,figsize = (8,8))

    im = ax.pcolormesh(X,Y,np.median(surfaces[surface_name]['values'], axis = 0), vmin = 160.,
                       vmax = 430., shading = 'auto')
    ax.scatter(x = surface_coords[:,0], y = surface_coords[:,1], s=8, c = mu_elev, vmin = 160., vmax = 430.,
               edgecolors = 'k', linewidths = 0.5)

    ax.set_xlabel('easting (m)')
    ax.set_ylabel('northing (m)')
    plt.colorbar(im)
    
    plt.savefig(r'E:\GA\GAB\Injune\{}_elevation.png'.format(surface_name), dpi = 200)
    plt.close('all')

# now we want to project our grids onto sections
lines = rj.data['line'][np.unique(rj.data['line_index'][df_interp['point_index'].values].data)]


grid_coords = np.column_stack((X.flatten(), Y.flatten()))

plot_kwargs = {'title': 'conductivity','max_depth': 300., 'vmin': 0.01,
                         'vmax': 1, 'cmap': 'viridis', 'ylabel': 'elevation \n (mAHD)',
                         'shade_doi': False, "log_plot": True}

colours = {'TopPrecipice': 'white', 'BasePrecipice': 'pink'}

for line in lines:

    infile =r'E:\GA\dash_data_Surat\section_data_rj\{}.pkl'.format(line)
    cond_xr = misc_utils.pickle2xarray(infile)
    line_coords = np.column_stack((cond_xr['easting'].values, cond_xr['northing'].values))
    # plot the section
    fig, ax = plt.subplots(1,1, figsize = (16,4))
    cond_section = plots.plot_grid(ax, cond_xr, 'conductivity_p50',
                                     panel_kwargs=plot_kwargs)
    for surface_name in ["TopPrecipice", "BasePrecipice"]:
        mask = df_interp['surface'] == surface_name
        coords = df_interp[mask][['X', 'Y']].values
        mu_elev = df_interp[mask]['DEM'].values - df_interp[mask]['p50'].values
        stdev = df_interp[mask]['depth_stdev'].values
        # now plot the interpreted points with error bars
        df_temp = df_interp[(df_interp['line'] == line) & (df_interp['surface'] == surface_name)]
        interp_coords = df_temp[['X', 'Y']].values
        dist, inds= spatial_functions.nearest_neighbours(interp_coords, line_coords, max_distance = 50)
        # plot on the map
        ax.errorbar(cond_xr['grid_distances'][inds], df_temp['DEM'].values - df_temp['p50'].values,
                    yerr=df_temp['depth_stdev'].values * 2.5, c=colours[surface_name], fmt=" ", barsabove=True)
        ax.scatter(cond_xr['grid_distances'][inds], df_temp['DEM'].values - df_temp['p50'].values,
                   c = colours[surface_name], label =  "{} picks".format(surface_name))

        # now we plot the petrel surface
        #dist, inds= spatial_functions.nearest_neighbours(line_coords, surfaces[surface_name]['petrel_coords'], max_distance = 500.)
        #inds = inds[np.isfinite(dist)]

        # plot as a line
        #ax.plot(cond_xr['grid_distances'].values, np.flipud(surfaces[surface_name]['petrel_values']).flatten()[inds],
        #         c =colours[surface_name], linestyle = 'dashed', label = 'petrel - {}'.format(surface_name))               


        # now model the surface at each point along the line

        n = 10000

        sectionX, sectionY = cond_xr['easting'].values, cond_xr['northing'].values
        # keep only potins within the convex hull
        keep_inds = []
        for i in range(len(sectionX)):
            if Point(sectionX[i], sectionY[i]).within(surfaces[surface_name]['convex_hull']):
                keep_inds.append(i)
        sectionX, sectionY = sectionX[keep_inds], sectionY[keep_inds]
        values = np.nan * np.ones(shape=(n, len(sectionX)), dtype=float)

        for i in range(n):

            v = idw(coords[:, 0], coords[:, 1], np.random.normal(mu_elev, stdev),
                         sectionX,sectionY,
                         power=2, max_distance= 10000.)
            values[i] = v

        median_values = np.median(values, axis = 0)
        # clip anything above groun level
        mask = np.where(median_values > cond_xr['elevation'].values[keep_inds])[0]
        median_values[mask] = np.nan
        p90 = np.quantile(values, 0.9, axis = 0)
        p90[mask] = np.nan
        p10 = np.quantile(values, 0.1, axis = 0)
        p10[mask] = np.nan
        if surface_name == 'TopPrecipice':
            ax.plot(cond_xr['grid_distances'][keep_inds].values,  median_values, c ='k', label = 'Top Precipice - median', linestyle = 'solid')
            ax.plot(cond_xr['grid_distances'][keep_inds].values, p10, c='grey', linestyle = "dashed", label = 'Top Precipice - p10')
            ax.plot(cond_xr['grid_distances'][keep_inds].values, p90, c='grey', linestyle = "dashed", label = 'Top Precipice - p90')
        else:
            ax.plot(cond_xr['grid_distances'][keep_inds].values,  median_values, c ='darkgrey', label = 'Base Precipice - median', linestyle = 'solid')
            ax.plot(cond_xr['grid_distances'][keep_inds].values, p10, c='grey', linestyle = "dashdot", label = 'Base Precipice - p10')
            ax.plot(cond_xr['grid_distances'][keep_inds].values, p90, c='grey', linestyle = "dashdot", label = 'Base Precipice - p90')

    plt.colorbar(cond_section)
    ax.set_xlabel('distance along line (m)')
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    plt.savefig(r'E:\GA\GAB\Injune\{}_section.png'.format(line), dpi=200)
    plt.close('all')
    gc.collect()
