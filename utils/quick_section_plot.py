import numpy as np
import netCDF4
from garjmcmctdem_utils import aem_utils
import matplotlib.pyplot as plt
from garjmcmctdem_utils import aem_utils, spatial_functions
from garjmcmctdem_utils import plotting_functions as plots

plot_kwargs = {'panel_1': {'color': 'black',
                         'ylabel': 'phid','legend': False},
             'panel_2': {'color': 'black',
                         'ylabel': 'tx_height','legend': False},
             'panel_3': {'color': 'blue',
                         'ylabel': 'txrx_dx','legend': False},
             'panel_4': {'color': 'blue',
                         'ylabel': 'txrx_dz','legend': False},
             'panel_5': {'ylabel': "asingh (fT)",'legend': False},
             'panel_6': {'ylabel': "asingh (fT)",'legend': False},
             'panel_7': {'title': 'galei conductivity','max_depth': 200., 'vmin': 0.001,
                         'vmax': 0.5, 'cmap': 'jet', 'ylabel': 'elevation \n (mAHD)',
                         'shade_doi': False}}


def galei_plot(line_number, plot_kwargs, save_fig = True, figsize = (12,30)):
    plt.close('all')

    fig = plt.figure(figsize = figsize)

    ax1 = fig.add_axes([0.1, 0.83, 0.8, 0.15])
    ax2 = fig.add_axes([0.1, 0.76, 0.8, 0.05])
    ax2.sharex(ax1)
    ax3 = fig.add_axes([0.1, 0.69, 0.8, 0.05])
    ax3.sharex(ax2)
    ax4 = fig.add_axes([0.1, 0.62, 0.8, 0.05])
    ax4.sharex(ax3)
    ax5 = fig.add_axes([0.1, 0.55, 0.8, 0.05])
    ax5.sharex(ax4)
    ax6 = fig.add_axes([0.1, 0.30, 0.8, 0.23])
    ax6.sharex(ax5)
    ax7 = fig.add_axes([0.1, 0.05, 0.8, 0.23])
    ax7.sharex(ax6)
    cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.2])

    grid_distances = galei.section_data[line_number]['grid_distances'].values

    ax1.plot(grid_distances, galei.section_data[line_number]['phid'],
             'k', label = 'PhiD')
    ax1.set_yscale('log')

    ax2.plot(em.section_data[line_number]['grid_distances'].values,
             em.section_data[line_number]['x_primary_field_observed'].values,
             color = 'k', label = 'x primary field observed')
    ax2.plot(em.section_data[line_number]['grid_distances'].values,
             em.section_data[line_number]['x_primary_field_predicted'].values,
             color = 'blue', label = 'x primary field predicted')
    ax3.plot(em.section_data[line_number]['grid_distances'].values,
             em.section_data[line_number]['z_primary_field_observed'].values,
             color = 'k', label = 'z primary field observed')
    ax3.plot(em.section_data[line_number]['grid_distances'].values,
             em.section_data[line_number]['z_primary_field_predicted'].values,
             color = 'blue', label = 'z primary field predicted')
    ax2.legend()
    ax3.legend()


    ax4.plot(grid_distances, galei.section_data[line_number]['inverted_txrx_dx'],
             color = 'k', label= 'inverted distance')
    ax4.plot([grid_distances[0],grid_distances[-1]],
             [galei.data['txrx_dx'][:], galei.data['txrx_dx'][:]],
             color = 'blue', label= 'nominal distance')
    ax4.set_ylabel('Dx (m)')
    ax4.legend()

    ax5.plot(grid_distances, galei.section_data[line_number]['inverted_txrx_dz'],
             color = 'k', label='inverted distance')
    ax5.plot([grid_distances[0],grid_distances[-1]],
             [galei.data['txrx_dz'][:], galei.data['txrx_dz'][:]],
             color = 'blue', label= 'nominal distance')
    ax5.set_ylabel('Dz (m)')
    ax5.legend()
    # plot the AEM data

    em_secondary_z_obs = em.section_data[line_number]['z_secondary_field_observed'].values
    em_secondary_z_pred = em.section_data[line_number]['z_secondary_field_predicted'].values

    em_secondary_x_obs = em.section_data[line_number]['x_secondary_field_observed'].values
    em_secondary_x_pred = em.section_data[line_number]['x_secondary_field_predicted'].values

    vector_sum_obs = np.sqrt(em_secondary_z_obs**2 + em_secondary_x_obs**2)
    vector_sum_pred = np.sqrt(em_secondary_z_pred ** 2 + em_secondary_x_pred ** 2)


    colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
               'grey', 'olive', 'cyan']

    for i in range(em_secondary_z_obs.shape[1]):
        ax6.plot(em.section_data[line_number]['grid_distances'],
                 np.arcsinh(vector_sum_obs[:,i]), 'k')
        ax6.plot(em.section_data[line_number]['grid_distances'],
                 np.arcsinh(vector_sum_pred[:,i]), color = colours[i],
                 linewidth = 0.5)


    ax6.set_ylabel('asinh (fT)')

    # PLot the section

    cond_section = plots.plot_grid(ax7, galei.section_data[line_number], 'conductivity',
                                   panel_kwargs=plot_kwargs['panel_7'])
    ax7.plot(grid_distances,
             galei.section_data[line_number]['tx_height'] + galei.section_data[line_number]['elevation'],
             color='blue', label='tx')
    ax7.legend()

    cb = fig.colorbar(cond_section, cax=cbar_ax, orientation='vertical')

    cb.ax.set_yticklabels([round(10 ** x, 4) for x in cb.get_ticks()])
    cb.set_label('conductivity (S/m)', fontsize=10)

    ax_array = np.array([ax1,ax2,ax3,ax4, ax5, ax6, ax7])
    return fig, ax_array

infile = "/home/nsymington/Documents/GA/AEM/Spectrem/nc/Musgraves_block1.nc"

galei = aem_utils.AEM_inversion(name = "galei",
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(infile))

lines = [11750, 11760, 11770, 11780, 11790, 11800, 11810, 11820,
         11830, 11840, 11850, 11860, 11870, 11880, 11890]


grid_vars = ['conductivity', 'phid', 'tx_height', #'inverted_tx_height',
             "inverted_txrx_dx", "inverted_txrx_dz"]

galei.grid_sections(variables = grid_vars, lines = lines,
                      xres = 10, yres = 4,
                      return_interpolated = True, save_to_disk = False)

em = aem_utils.AEM_data(system_name = 'spectrem',
                        netcdf_dataset= netCDF4.Dataset(infile))

grid_vars = ['x_primary_field_observed', 'x_secondary_field_observed', 'z_primary_field_observed',
                   'z_secondary_field_observed', 'x_primary_field_predicted', 'x_secondary_field_predicted',
                   'z_primary_field_predicted', 'z_secondary_field_predicted']

em.griddify_variables(grid_vars, lines,  return_gridded = True, save_to_disk = False)

for line in lines:
    fig, ax_array = galei_plot(line, plot_kwargs = plot_kwargs, figsize=(40,10))
    plt.savefig('Spectrem_block1_{}.png'.format(line), dpi = 200)
    plt.close('all')
plt.show()



