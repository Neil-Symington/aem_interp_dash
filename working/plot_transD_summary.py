import numpy as np
import matplotlib.pyplot as plt
import os, glob
from garjmcmctdem_utils import spatial_functions, aem_utils
from garjmcmctdem_utils import plotting_functions as plots
import netCDF4
import gc

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

    cp_pdf = cp_freq / freq.sum(axis=1)[0]


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


    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'depth_cells': depth_cells,
            'cond_cells': cond_cells, 'det_cond': det_cond, 'det_depth_top': det_depth_top, 'det_doi': det_doi,
            'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial,
            'elevation': elevation, 'n_histogram_samples': n_hist_samples}


nc_infile = r"E:\GA\dash_data_Surat\Injune_pmaps_reduced_concatenated.nc"
rj = aem_utils.AEM_inversion(name = 'garj', inversion_type='stochastic',
                                netcdf_dataset=netCDF4.Dataset(nc_infile))

nc_infile = r"E:\GA\dash_data_Surat\Injune_lci_MGA55.nc"
lci = aem_utils.AEM_inversion(name = 'lci', inversion_type='deterministic',
                                netcdf_dataset=netCDF4.Dataset(nc_infile))


indir = r"C:\Users\u77932\Documents\GAB\data\AEM\transD"

# Now we will extract the data into dictionaries
trans_D = {}

# get our lines
for file in glob.glob(os.path.join(indir, "*_summary_xyzrho.txt")):
    line = int(file.split("_")[3])
    if not line in trans_D.keys():
        trans_D[line] = {}
    # Add the depth top array
    trans_D[line]["depth_top"] = np.loadtxt(os.path.join(indir, "depth_tops.txt"))

# Now we extract the data from the output file one a line by line basis

for line in trans_D.keys():
    for item in ['low', 'mid', 'hi', 'avg']:
        arr = np.loadtxt(os.path.join(indir, "rho_{}_line_{}_summary_xyzrho.txt".format(item, line)))
        coords = np.unique(arr[:,:2], axis = 0)
        if "coords" not in trans_D[line].keys():
            trans_D[line]["coords"] = coords
        # now get the data
        rho = np.loadtxt(os.path.join(indir, "rho_{}_line_{}_summary.txt".format(item, line)))
        sigma = 1./(10.**rho)
        trans_D[line]["{}_sigma".format(item)] = sigma
    # get misfit and ddz
    for item in ["sdev", "mean"]:
        arr = np.loadtxt(os.path.join(indir, "phid_{}_line_{}_summary.txt".format(item, line)))
        trans_D[line]["phid_{}".format(item)] = arr
        arr = np.loadtxt(os.path.join(indir, "ddz_{}_line_{}_summary.txt".format(item, line)))
        trans_D[line]["ddz_{}".format(item)] = arr
    # now find the fiducial using a nearest neighbour search
    dist, inds = spatial_functions.nearest_neighbours(trans_D[line]['coords'],rj.coords, max_distance=20.)
    trans_D[line]['fiducials'] = rj.data['fiducial'][inds].data

# now we produce our plots

lines = trans_D.keys()



for line in lines:
    for i, fid in enumerate(trans_D[line]['fiducials']):
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize = (16,8), sharey = True)
        point_index = np.where(fid == rj.data['fiducial'][:])[0][0]
        colours = {'p10': 'blue', 'p50': 'orange', 'p90': 'green'}
        # Extract the data from the netcdf data
        D = extract_rj_sounding(rj,lci, point_index)
        pmap = D['conductivity_pdf']
        x1,x2,y1,y2 = D['conductivity_extent']
        n_depth_cells, n_cond_cells  = pmap.shape
        x = 10**np.linspace(x1,x2, n_cond_cells)
        y = np.linspace(y2,y1, n_depth_cells)

        # plot the pmap and quantile models
        # ax1
        im = ax1.imshow(D['conductivity_pdf'], extent=D['conductivity_extent'],
                        aspect='auto', cmap='rainbow')

        #  PLot the median, and percentile plots
        ax1.plot(np.log10(D['cond_p10']), D['depth_cells'], c='k', linestyle='dashed', label='p10')
        ax1.plot(np.log10(D['cond_p90']), D['depth_cells'], c='k', linestyle='dashed', label='p90')
        ax1.plot(np.log10(D['cond_p50']), D['depth_cells'], c='k', label='p50')

        ax1.set_title('rj-MCMC probability map')
        ax1.set_ylabel('depth (mBGL)')
        ax1.set_xlabel('log10 Conductivity (S/m)')
        ax1.grid(which='both')

        ax1.set_ylim(400., 0.)
        ax1.set_xlim(-3,0)
        ax1.legend()
        min_phid = rj.data['misfit_lowest'][point_index]
        ax1.text(x = -3, y = 100., s = "Min phiD =\n {}".format(np.round(min_phid,2)))

        # Ax 2
        ax2.plot(D['change_point_pdf'], D['depth_cells'], label='P(change point)')
        ax2.set_ylim(ax1.get_ylim())
        #ax2.set_yticks(np.arange(0, 500, 20.))
        ax2.set_title('change point probability')
        ax2.set_ylim(ax1.get_ylim())

        ax2.grid(which='both')

        # ax3
        ax3.plot(np.log10(trans_D[line]['low_sigma'][:,i]), trans_D[line]['depth_top'], c='k', linestyle='dashed', label='p10')
        ax3.plot(np.log10(trans_D[line]['mid_sigma'][:,i]), trans_D[line]['depth_top'], c='k', linestyle='solid', label='p50')
        ax3.plot(np.log10(trans_D[line]['hi_sigma'][:,i]), trans_D[line]['depth_top'], c='k', linestyle='dashed', label='p90')

        ax3.set_title('transD quantiles')
        ax3.set_ylabel('depth (mBGL)')
        ax3.set_xlabel('log10 Conductivity (S/m)')
        ax3.grid(which='both')
        ax3.set_xlim(-3,0)

        ax3.legend()
        # get phid
        phid = trans_D[line]['phid_mean'][i]
        ax3.text(x = -3, y = 100., s = "Mean phiD =\n {}".format(np.round(phid,2)))

        # ax4 
        ax4.plot(trans_D[line]['ddz_mean'][:,i], trans_D[line]['depth_top'], c='k', linestyle='solid', label='mean')
        ax4.plot(trans_D[line]['ddz_mean'][:,i] - trans_D[line]['ddz_sdev'][:,i], trans_D[line]['depth_top'], c='k', linestyle='dashed')
        ax4.plot(trans_D[line]['ddz_mean'][:,i] + trans_D[line]['ddz_sdev'][:,i], trans_D[line]['depth_top'], c='k', linestyle='dashed')
        ax4.set_xlabel('ddz')

        plt.savefig(r"C:\Users\u77932\Documents\GAB\data\AEM\comparison_plots\line_{}_fid_{}_comparison_plot.png".format(line, fid))
        plt.close('all')
        gc.collect()





