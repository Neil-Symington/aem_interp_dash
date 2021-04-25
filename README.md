# garjmcmctdem_utils

This branch is the code base for running a dash app for visualising and interpreting
airborne electromagnetic data. A feature of this app is that it is able to create 
visualisations of a bulk conductivity model ensemble generated using a stochastic AEM 
inversion code. The plots that are generated with this app are interactive and can be used 
for interpreting stratigraphic features with predictive uncertainty.

## Installation

This package has many dependencies and we use anaconda to manage them. The following conda
commands create an environment that should run the app no problem.

#### Create an environment
>conda create -n AEM_interp python=3.7
>conda activate AEM_interp

#### install required packages
>conda install -c conda-forge numpy pandas netCDF4 shapely geopandas matplotlib scipy dash
>conda install -c conda-forge xarray dask bottleneck
>conda install -c scikit-learn
>conda install -c plotly
>pip install affine
>pip install pyyaml

#### install garjmcmctdem
>pip install -e <path to garjmcmctdem_utils>

## Usage

The first step in running the dash app is to prepare the data and point the app to it. We
assume a user has already run AEM inversions using both a deterministic and/ or stochastic
code. We use contractor ga_aem (https://github.com/GeoscienceAustralia/ga-aem) for 
running these inversions on the National Computational Infrastructure (NCI).

NetCDF is a convenient format for storing and handling large point, grid or line data. To 
convert our AEM line data to netcdf we use the geophys utils package developed at GA: 
 https://github.com/GeoscienceAustralia/geophys_utils/tree/master/geophys_utils .
This package itself has a host of dependencies and we use a seperate conda environment.
Scripts for converting the EM data and deterministic inversion can be found in the 
"utils/conversion" directory.

The probabilistic inversions are handled differently. AEM soundings are inverted on a point
by point basis, with each inversion creating a single netcdf file. The 
"utils/conversion/concatenate_netcdf_file.py" script demonstrates how to compile the netcdf
files into a single file containing just the important information. We rely very heavily 
on consistent field names for this app so we highly recommend a user familiarising
themselves with the  "utils/conversion/netcdf_settings.yaml" file.

Once all the data are prepared, they should be copied into a single directory. Use the
"workflow/interpretation_config.yaml" file to set the path to this directory. To avoid 
expensive gridding of AEM line data onto sections/ grids, we preprocess the data and save
the 'rasterised' sections as xarray files. The gridding parameters and path are set using
the interpretation_config.yanl file. More information on how this works can be found in
the wiki.

Once the data is prepared and the settings are satisfactory, run the app (app.py) 
using thw python from the conda environment. If gridding sections, the app may take a
 while to run. Once the preprocessing is done, the terminal should return a message 
 similar to the following:
 
 Dash is running on http://127.0.0.1:8050/

 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)

If all has worked, the app can be opened by pasting the url into your browser (we have
tested on firefox and chrome). Happy interpreting :)


