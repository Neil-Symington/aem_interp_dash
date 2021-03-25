'''
This script generates high quality pmap plots for some key rj fiducials. This is for supporting the Surat interpretation

Neil Symington
neil.symington@ga.gov.au
'''

import numpy as np
from garjmcmctdem_utils import spatial_functions, aem_utils, plotting_functions
import netCDF4
from garjmcmctdem_utils.misc_utils import pickle2xarray
import matplotlib.pyplot as plt

# Bring int he lci

lci = aem_utils.AEM_inversion(name = "lci", inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset("/home/nsymington/Documents/GA/dash_data_Surat/Injune_lci_MGA55.nc"))
lci.load_lci_layer_grids_from_pickle("/home/nsymington/Documents/GA/dash_data_Surat/Injune_layer_grids.p")

# Create polylines
lci.create_flightline_polylines(crs = 'EPSG:28355')

lines = lci.data['line'][:]

gdf_lines = lci.flightlines[np.isin(lci.flightlines['lineNumber'], lines)]


# bring in the rj inversion
nc_file = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_pmaps_reduced_concatenated.nc"

rj = aem_utils.AEM_inversion(name = 'garjmcmctdem',  inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(nc_file))

# get a list of the indices to plot

inds = [4319, 3635, 5530, 1077, 5071, 4944, # from line 201001
        3418, 4361, 293, 2784, 1496] # from line 200401
# get a list of the fids

fids = rj.data['fiducial'][inds].data

rj = None

# now we have to go into the original pmap file because they contain more information

infile  = "/home/nsymington/Documents/GA/dash_data_Surat/Injune_rjmcmc_pmaps.nc"
rj = aem_utils.AEM_inversion(name = 'garjmcmctdem',  inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(infile))

new_inds = np.where(np.isin(rj.data['fiducial'][:].data, fids))[0]


def extract_rj_sounding(rj, point_index=0):

    rj_dat = rj.data

    n_hist_samples = rj_dat['log10conductivity_histogram'][point_index].data.sum(axis = 1)[0]

    freq = rj_dat['log10conductivity_histogram'][point_index].data.astype(np.float)

    easting = np.float(rj_dat['easting'][point_index].data)
    northing = np.float(rj_dat['northing'][point_index].data)

    cond_pdf = freq / freq.sum(axis=1)[0]

    cond_pdf[cond_pdf == 0] = np.nan

    cp_freq = rj_dat["interface_depth_histogram"][point_index].data.astype(np.float)

    cp_pdf = cp_freq / freq.sum(axis=1)[0]

    cond_cells = rj_dat['cond_bin_centre'][:]

    depth_cells = rj_dat['layer_centre_depth'][:]

    extent = [cond_cells.min(), cond_cells.max(), depth_cells.max(), depth_cells.min()]

    p10 = np.power(10, rj_dat['conductivity_p10'][point_index].data)
    p50 = np.power(10, rj_dat['conductivity_p50'][point_index].data)
    p90 = np.power(10, rj_dat['conductivity_p90'][point_index].data)

    # get line under new schema
    line_index = int(rj_dat['line_index'][point_index])
    line = int(rj_dat['line'][line_index])
    fiducial = float(rj_dat['fiducial'][point_index])
    elevation = rj_dat['elevation'][point_index]

    misfit = rj_dat['misfit'][point_index]

    nlayers_histogram = rj_dat['nlayers_histogram'][point_index]

    # get the section data
    infile = '/home/nsymington/Documents/GA/dash_data_Surat/section_data_rj/{}.pkl'.format(line)

    xarr = pickle2xarray(infile)

    dist = spatial_functions.xy_2_var(xarr, np.array([[easting, northing]]), 'grid_distances')
    section_misfit = xarr['misfit_lowest'].values
    section_p50 = xarr['conductivity_p50'].values
    grid_distances = xarr['grid_distances']
    grid_elevations = xarr['grid_elevations']

    burn_in = rj_dat['nburnin'][:].data
    nsamples = rj_dat.nsamples
    ndata = rj_dat.dimensions['data'].size
    sample_no = 1 + np.arange(0, rj_dat.dimensions['convergence_sample'].size) * int(np.ceil(rj_dat['nsamples'][:]/rj_dat.dimensions['convergence_sample'].size))

    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'depth_cells': depth_cells, "nlayers_histogram": nlayers_histogram,
            'cond_cells': cond_cells, "rj_dist": dist, "misfit": misfit, "burnin": burn_in, 'sample_no': sample_no,
            'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial, 'ndata': ndata,
            'elevation': elevation, 'n_histogram_samples': n_hist_samples, 'nsamples': nsamples, 'section_misfit': section_misfit,
            "section_p50": section_p50, "grid_distances": grid_distances, "grid_elevations": grid_elevations}


def pmap_plot(D, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_axes([0.05, 0.35, 0.35, 0.62])
    ax2 = fig.add_axes([0.45, 0.35, 0.15, 0.62])
    ax3 = fig.add_axes([0.70, 0.7, 0.25, 0.25])
    ax4 = fig.add_axes([0.70, 0.35, 0.25, 0.25])
    ax5 = fig.add_axes([0.1, 0.18, 0.76, 0.03])
    ax6 = fig.add_axes([0.1, 0.05, 0.76, 0.13])
    #ax
    cbar_ax1 = fig.add_axes([0.05, 0.29, 0.35, 0.01])
    cbar_ax2 = fig.add_axes([0.88, 0.05, 0.01, 0.2])

    # Plot probability map

    # ax1
    im = ax1.imshow(D['conductivity_pdf'], extent=D['conductivity_extent'],
                    aspect='auto', cmap='rainbow')

    #  PLot the median, and percentile plots
    ax1.plot(np.log10(D['cond_p10']), D['depth_cells'], c='k', linestyle='dashed', label='p10')
    ax1.plot(np.log10(D['cond_p90']), D['depth_cells'], c='k', linestyle='dashed', label='p90')
    ax1.plot(np.log10(D['cond_p50']), D['depth_cells'], c='k', label='p50')

    ax1.set_xticklabels([round(10 ** float(x), 4) for x in ax1.get_xticks()])

    ax1.set_title('rj-MCMC probability map')
    ax1.set_ylabel('depth (mBGL)')
    ax1.set_xlabel('Conductivity (S/m)')
    ax1.grid(which='both')

    ax1.set_ylim(150., 0.)
    ax1.legend()

    # Ax 2
    ax2.plot(D['change_point_pdf'], D['depth_cells'], label='P(change point)')
    ax2.set_ylim(ax1.get_ylim())
    #ax2.set_yticks(np.arange(0, 500, 20.))
    ax2.set_title('change point probability')
    ax2.set_ylim(ax1.get_ylim())

    ax2.grid(which='both')

    # Add the misfit
    for i in range(D['misfit'].shape[0]):
        misfits = D['misfit'][i]
        ax3.plot(D['sample_no'], misfits / D['ndata'])

    ax3.plot([1, D['nsamples']], [1, 1], 'k')
    ax3.plot([D['burnin'], D['burnin']], [0.01, 1e4], 'k')
    ax3.set_xlim([1, D['sample_no'].max()])
    ax3.set_ylim(0.1, ax3.get_ylim()[1])

    ax3.set_xscale('log')
    ax3.set_yscale('log')

    ax3.set_xlabel("sample #")
    ax3.set_ylabel("Normalised misfit")

    # ax 4 will show the location of the point on a map
    # flight lines
    layer = 5
    cond_grid = np.log10(lci.layer_grids['Layer_{}'.format(layer)]['conductivity'])

    im4 = ax4.imshow(cond_grid, extent=lci.layer_grids['bounds'],
                     cmap='viridis',
                     vmin=np.log10(0.001),
                     vmax=np.log10(1.))

    buffer = 5000.

    ax4.set_xlim(D['easting'] - buffer,
                 D['easting'] + buffer)
    ax4.set_ylim(D['northing'] - buffer,
                 D['northing'] + buffer)
    # now plot the lines
    for linestring, lineNo in zip(gdf_lines.geometry, gdf_lines.lineNumber):
        if lineNo == D['line']:
            c = 'red'
        else:
            c = 'k'
        x, y = linestring.xy
        ax4.plot(x,y, c = c)
    ax4.plot(D['easting'], D['northing'], 'x', c='k')

    ax4.set_title('LCI layer slice {}'.format(layer), fontsize=10)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    ax4.tick_params(axis='both', which='minor', labelsize=8)

    ax5.plot(D['grid_distances'], D['section_misfit'], 'k')

    ax5.set_title('GARJMCMCTDEM  conductivity P50 section - ' + str(D['line']))

    # Ax 6
    im6 = ax6.imshow(np.log10(D["section_p50"]),
                     extent=(D['grid_distances'].min(), D['grid_distances'].max(),
                             D['grid_elevations'].min(),D['grid_elevations'].max()),
                     cmap = 'viridis', vmin = -3, vmax = 0,
                     aspect = 'auto')
    ax6.plot([D['rj_dist'], D['rj_dist']], [D['grid_elevations'].min(),D['grid_elevations'].max()], c = 'grey')
    ax6.set_xlim(D['rj_dist'] - 500., D['rj_dist'] + 500.)
    ax5.set_xlim(ax6.get_xlim())
    ax6.set_ylim(250., 450.)

    # cbar axes
    cb1 = fig.colorbar(im, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('probabilitiy', fontsize=10)

    cb2 = fig.colorbar(im6, cax=cbar_ax2, orientation='vertical')

    cb2.ax.set_yticklabels([round(10 ** x, 4) for x in cb2.get_ticks()])
    cb2.set_label('conductivity (S/m)', fontsize=10)

    ax_array = np.array([ax1, ax2, ax3, ax4, ax5, ax6,])

    return fig, ax_array

def generate_pmap_plot(point_index):

    D = extract_rj_sounding(rj, point_index)

    fig, ax_array = pmap_plot(D, figsize=(8, 8))
    plt.savefig("pmAP_plot_{}.png".format(point_index))

    plt.close('all')

for ind in new_inds:
    generate_pmap_plot(ind)