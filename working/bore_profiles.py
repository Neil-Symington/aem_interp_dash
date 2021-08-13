import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from garjmcmctdem_utils import spatial_functions, aem_utils
from garjmcmctdem_utils import plotting_functions as plots
import matplotlib.patches as patches
import netCDF4
from garjmcmctdem_utils.misc_utils import pickle2xarray

def extract_rj_sounding(rj, det, point_index=0):

    rj_dat = rj.data
    det_dat = det.data

    n_hist_samples = rj_dat['log10conductivity_histogram'][point_index].data.sum(axis = 1)[0]

    freq = rj_dat['log10conductivity_histogram'][point_index].data.astype(np.float)

    easting = np.float(rj_dat['easting'][point_index].data)
    northing = np.float(rj_dat['northing'][point_index].data)

    cond_pdf = freq / freq.sum(axis=1)[0]

    cond_pdf[cond_pdf == 0] = np.nan

    cp_freq = rj_dat["interface_depth_histogram"][point_index].data.astype(np.float)

    cond_cells = rj_dat['conductivity_cells'][:]

    depth_cells = rj_dat['layer_centre_depth'][:]

    extent = [cond_cells.min(), cond_cells.max(), depth_cells.max(), depth_cells.min()]

    p10 = np.power(10, rj_dat['conductivity_p10'][point_index].data)
    p50 = np.power(10, rj_dat['conductivity_p50'][point_index].data)
    p90 = np.power(10, rj_dat['conductivity_p90'][point_index].data)

    distances, indices = spatial_functions.nearest_neighbours([easting, northing],
                                                              det.coords,
                                                              max_distance=100.)
    point_ind_det = indices[0]

    det_cond = det_dat['conductivity'][point_ind_det].data
    det_depth_top = det_dat['layer_top_depth'][point_ind_det].data

    det_doi = det_dat['depth_of_investigation'][point_ind_det].data

    # get line under new schema
    line_index = int(rj_dat['line_index'][point_index])
    line = int(rj_dat['line'][line_index])
    fiducial = float(rj_dat['fiducial'][point_index])
    elevation = rj_dat['elevation'][point_index]

    return {'conductivity_pdf': cond_pdf, "change_point_freq": cp_freq, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'depth_cells': depth_cells,
            'cond_cells': cond_cells, 'det_cond': det_cond, 'det_depth_top': det_depth_top, 'det_doi': det_doi,
            'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial,
            'elevation': elevation, 'n_histogram_samples': n_hist_samples}



infile = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_strat_proximal_to_AEM_rj.csv"

df = pd.read_csv(infile)

nc_infile = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_pmaps_reduced_concatenated.nc"

rj = aem_utils.AEM_inversion(name = 'rj', inversion_type = 'stochastic', netcdf_dataset = netCDF4.Dataset(nc_infile))

nc_infile = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_lci_MGA55.nc"

lci = aem_utils.AEM_inversion(name = 'lci', inversion_type = 'deterministic',
                                    netcdf_dataset = netCDF4.Dataset(nc_infile))


wells = df['BORE_NAME'].unique()

for well in wells:
    df_temp = df[df['BORE_NAME'] == well]
    depth_top = df_temp['MD'].values
    depth_bottom = np.ones(shape = depth_top.shape, dtype = depth_top.dtype) * df_temp['TD'].unique()[0]
    depth_bottom[:-1] = depth_top[1:]
    units = df_temp['GA_UNIT'].values
    fid = df_temp['fiducial'].unique()
    assert len(fid) == 1
    point_index = np.where(rj.data['fiducial'] == fid)[0][0]
    D = extract_rj_sounding(rj, lci, point_index)
    fig, ax1 = plt.subplots(1,1, figsize = (8,8))
    # ax1 will be the pmaps
    im = ax1.imshow(D['conductivity_pdf'], extent=D['conductivity_extent'],
                    aspect='auto', cmap='Greys')

    #  PLot the median, and percentile plots
    colours = {'p10': 'blue', 'p50': 'orange', 'p90': 'green'}
    for item in ['p10', 'p50', 'p90']:
        cond_expanded, depth_expanded = plots.profile2layer_plot(D['cond_{}'.format(item)],
                                                                 D['depth_cells'] - 1)
        ax1.plot(np.log10(cond_expanded), depth_expanded,
                 c=colours[item], label=item)

    lci_expanded, depth_expanded = plots.profile2layer_plot(D['det_cond'], D['det_depth_top'])

    ax1.plot(np.log10(lci_expanded), depth_expanded, c='pink',
             linestyle='dashed', label='lci')


    ax1.set_xticklabels([round(10 ** float(x), 4) for x in ax1.get_xticks()])

    ax1.set_title('rj-MCMC probability map')
    ax1.set_ylabel('depth (mBGL)')
    ax1.set_xlabel('Conductivity (S/m)')
    ax1.set_yticks(np.arange(0,400, 20))
    ax1.grid(which='both')
    #ax1.legend()
    # now plot ax 2
    unit_colours = {'Hutton Sandstone': 'blue', 'Evergreen Formation': 'green',
                    'Precipice Sandstone': 'lightskyblue', 'Rewan Formation': 'pink',
                    'Bandanna Formation': 'orange', 'Boxvale Sandstone Member': 'purple',
                    'Black Alley Shale': 'darkgrey', 'Mantuan Productus Bed': 'lightgrey',
                    'Peawaddy Formation': 'coral', 'Catherine Sandstone': 'pink',
                    'Ingelara Formation': 'red', 'Freitag Formation': 'blanchedalmond',
       'Aldebaran Sandstone': 'yellow', 'Cattle Creek Formation': 'cyan',
                    'Reids Dome Beds': 'slategray', 'Moolayember Formation': 'salmon',
                    'Clematis Group': 'plum'}
    for i in range(len(units)):
        # Create a Rectangle patch
        if depth_top[i] > 400.:
            break
        thickness = depth_bottom[i] - depth_top[i]
        rect = patches.Rectangle((0., depth_top[i]),
                                 0.25, thickness, linewidth=1,
                                 edgecolor='k', facecolor=unit_colours[units[i]],
                                 label = units[i])

        # Add the patch to the Axes
        ax1.add_patch(rect)
        ax1.legend(loc = 4)
    # get the distance
    [x1, y1] = df_temp[['easting', 'northing']].values[0]
    x2 = rj.data['easting'][point_index].data
    y2 = rj.data['northing'][point_index].data


    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    ax1.text(x = -1, y = -10, s=  "distance: {}m".format(np.round(distance, 1)))

    plt.savefig("/home/nsymington/Documents/temp/injune_bores/{}.png".format(well))
    #plt.show()