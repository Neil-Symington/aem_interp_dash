import numpy as np
import netCDF4
from garjmcmctdem_utils import aem_utils
import matplotlib.pyplot as plt
from garjmcmctdem_utils import aem_utils, spatial_functions

infile = "/home/nsymington/Documents/GA/AEM/Spectrem/nc/Musgraves_block1_flight7.nc"

galei = aem_utils.AEM_inversion(name = "galei",
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(infile))

# cretae a histogram of x and z component primary field and phid for each flight

# fig, ax_array = plt.subplots(13,3, sharex = 'col')
#
# for i in range(13):
#     inds = np.where(galei.data['flight_index'][:] == i)[0]
#     ax_array[i,0].hist(galei.data['x_primary_field_observed'][inds])
#     ax_array[i, 1].hist(galei.data['z_primary_field_observed'][inds])
#     ax_array[i, 2].hist(np.log10(galei.data['phid'][inds]))
#
# plt.show()

# fig, ax_array = plt.subplots(7,2, sharex = 'col', figsize = (8,24))
#
# for integer in range(13):
#     i = np.floor(integer/2).astype(int)
#     if integer%2 == 0:
#         j  = 0
#
#     else:
#         j = 1
#     flight_number = galei.data['flight'][integer]
#     inds = np.where(galei.data['flight_index'][:] == i)[0]
#     ax_array[i,j].scatter(np.log10(galei.data['phid'][inds]),
#                         galei.data['z_primary_field_observed'][inds], c = 'blue')
#     ax_array[i,j].set_title('flight number {}'.format(flight_number))
# ax_array[-1, 0].set_xlabel('log10 (phid)')
# ax_array[-1, 1].set_xlabel('log10 (phid)')
# ylabel = 'z primary \nfield observed'
# ax_array[0,0].set_ylabel(ylabel)
# ax_array[1,0].set_ylabel(ylabel)
# ax_array[2,0].set_ylabel(ylabel)
# ax_array[3,0].set_ylabel(ylabel)
# ax_array[4,0].set_ylabel(ylabel)
# ax_array[5,0].set_ylabel(ylabel)
# ax_array[6,0].set_ylabel(ylabel)
# plt.show()
# fig, ax_array = plt.subplots(7,2, sharex = 'col',sharey = True, figsize = (8,32))
#
# for integer in range(13):
#     i = np.floor(integer/2).astype(int)
#     if integer%2 == 0:
#         j  = 0
#
#     else:
#         j = 1
#     flight_number = galei.data['flight'][integer]
#     inds = np.where(galei.data['flight_index'][:] == integer)[0]
#     ax_array[i,j].scatter(galei.data['x_primary_field_observed'][inds],
#                         galei.data['x_primary_field_predicted'][inds],
#                           c = np.log10(galei.data['phid'][inds]))
#     #ax_array[i,j].scatter(galei.data['z_primary_field_observed'][inds],
#     #                      galei.data['z_primary_field_predicted'][inds],
#     #    c = np.log10(galei.data['phid'][inds]))
#
#     ax_array[i,j].set_title('flight number {}'.format(flight_number))
# xlabel = 'x_primary \nfield predicted'
# ax_array[-1, 0].set_xlabel(xlabel)
# ax_array[-1, 1].set_xlabel(xlabel)
# ylabel = 'inverted pitch'
# ax_array[0,0].set_ylabel(ylabel)
# ax_array[1,0].set_ylabel(ylabel)
# ax_array[2,0].set_ylabel(ylabel)
# ax_array[3,0].set_ylabel(ylabel)
# ax_array[4,0].set_ylabel(ylabel)
# ax_array[5,0].set_ylabel(ylabel)
# ax_array[6,0].set_ylabel(ylabel)
# plt.show()
fig, ax = plt.subplots(1, figsize = (4,4))

ax.scatter(galei.data['x_primary_field_observed'][:],
                        galei.data['x_primary_field_predicted'][:],
                          c = np.log10(galei.data['phid'][:]))

plt.show()