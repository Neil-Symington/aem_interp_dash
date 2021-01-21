import netCDF4
import os
import numpy as np
import matplotlib.pyplot as plt
from garjmcmctdem_utils import spatial_functions
from garjmcmctdem_utils import plotting_functions as plots


# Bring in the data
d = netCDF4.Dataset(r"C:\Users\u77932\Documents\EFTF2\SW\working\synthetics\hydrostrat_resolvability_testing\output\synthetics_rjmcmc_pmaps.nc")


def extract_soundings(rj_dat, point_index):
    freq = rj_dat['log10conductivity_histogram'][point_index].data.astype(np.float)

    cond_pdf = freq / freq.sum(axis=1)[0]

    cond_pdf[cond_pdf == 0] = np.nan

    cp_freq = rj_dat["interface_depth_histogram"][point_index].data.astype(np.float)

    cp_pdf = cp_freq / freq.sum(axis=1)[0]

    laybins = rj_dat['nlayers_histogram'][point_index].data

    lay_prob = laybins / freq.sum(axis=1)[0]

    condmin, condmax = rj_dat.min_log10_conductivity, rj_dat.max_log10_conductivity

    ncond_cells = rj_dat.dimensions['conductivity_cells'].size

    cond_cells = np.linspace(condmin, condmax, ncond_cells)

    pmin, pmax = rj_dat.min_depth, rj_dat.max_depth

    depth_cells = rj_dat['layer_centre_depth'][:]

    extent = [cond_cells.min(), cond_cells.max(), depth_cells.max(), depth_cells.min()]

    mean = np.power(10, rj_dat['conductivity_mean'][point_index].data)
    p10 = np.power(10, rj_dat['conductivity_p10'][point_index].data)
    p50 = np.power(10, rj_dat['conductivity_p50'][point_index].data)
    p90 = np.power(10, rj_dat['conductivity_p90'][point_index].data)

    try:
        misfit = rj_dat['misfit'][point_index].data
    except IndexError:
        misfit = None
    try:
        sample_no = np.arange(1, rj_dat.dimensions['convergence_sample'].size + 1)
    except KeyError:
        sample_no = None


    burnin = rj_dat.nburnin
    nsamples = rj_dat.nsamples

    nchains = rj_dat.nchains

    ## TODO add a function for 'expanding' the true model

    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'cond_mean': mean, 'depth_cells': depth_cells,
            'nlayer_bins': laybins, 'nlayer_prob': lay_prob, 'nsamples': nsamples,
            'ndata': rj_dat.dimensions['data'].size,
            "nchains": nchains, 'burnin': burnin, 'misfit': misfit, 'sample_no': sample_no, 'cond_cells': cond_cells}

def plot_pmaps(D, true_model, outfile):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8,8))
    im = ax1.imshow(D['conductivity_pdf'], extent=D['conductivity_extent'],
                    aspect='auto', cmap='rainbow')

    #  PLot the median, and percentile plots
    ax1.plot(np.log10(D['cond_p10']), D['depth_cells'], c='k', linestyle='dashed', label='p10')
    ax1.plot(np.log10(D['cond_p90']), D['depth_cells'], c='k', linestyle='dashed', label='p90')
    #ax1.plot(np.log10(D['cond_p50']), D['depth_cells'], c='k', label='p50')
    #ax1.plot(np.log10(D['cond_mean']), D['depth_cells'], c='grey', label='mean')
    # plot the true model

    cond_expanded, depth_expanded = plots.profile2layer_plot(true_model['conductivity'],
                                                             true_model['depth_tops'])
    ax1.plot(np.log10(cond_expanded), depth_expanded, c='black', label='true model')


    ax1.set_xlim([-3.5, 0.])
    ax1.set_xticklabels([round(10 ** float(x), 4) for x in ax1.get_xticks()])

    ax1.set_title('rj-MCMC probability map')
    ax1.set_ylabel('depth (mBGL)')
    ax1.set_xlabel('Conductivity (S/m)')
    ax1.grid(which='both')

    ax1.set_ylim(200,0)


    ax1.legend(loc=3)

    # Ax 2
    ax2.plot(D['change_point_pdf'], D['depth_cells'], label='P(change point)')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_yticks(np.arange(0, 500, 20.))
    ax2.set_title('change point probability')
    ax2.set_ylim(ax1.get_ylim())

    ax2.legend()
    ax2.grid(which='both')

    plt.savefig(outfile)

def parse_synthetic_models(infile):
    a = np.loadtxt(infile, skiprows=1, delimiter=',')

    # Create a dictionary with our results
    models = {}
    for row in a:
        key = row[0]
        thickness = row[1:4]
        depth_tops = spatial_functions.thickness_to_depth(thickness)
        cond = row[4:]
        models[key] = {'depth_tops': depth_tops, 'conductivity': cond}
    return models

synthetic_model_file= r"C:\Users\u77932\Documents\EFTF2\SW\working\synthetics\hydrostrat_resolvability_testing\synthetic_models.csv"

true_models = parse_synthetic_models(synthetic_model_file)

outdir = r"C:\Users\u77932\Documents\EFTF2\SW\working\synthetics\hydrostrat_resolvability_testing\figs"
for i in range(d.dimensions['point'].size):
    D = extract_soundings(d, i)
    outfile = os.path.join(outdir, "synthetic_{}_pmap.png".format(str(i)))
    true_model = true_models[i]
    plot_pmaps(D, true_model, outfile)