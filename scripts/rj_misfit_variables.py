import numpy as np
import netCDF4

infile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\rjmcmc\Injune_petrel_rjmcmc_pmaps.nc"

d = netCDF4.Dataset(infile, 'a')

# get the misfit

misfit = d['misfit'][:]/d.dimensions['data'].size

# add the minimum and average misfit to the datafile

min_misfit = d.createVariable("misfit_lowest","f8",('point'))
#min_misfit = d['misfit_lowest']
min_misfit[:] = np.min(misfit, axis = (1,2))
min_misfit.aseg_gdf_format = 'E10.6'
min_misfit.long_name = 'Lowest misfit on any chain'


ave_misfit = d.createVariable("misfit_average","f8",('point'))
#ave_misfit = d['misfit_average']
ave_misfit[:] = np.median(misfit, axis = (1,2))
ave_misfit.aseg_gdf_format = 'E10.6'
ave_misfit.long_name = 'Median misfit over all chains'

d.close()