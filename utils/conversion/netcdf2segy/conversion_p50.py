
#!/usr/bin/env python

# ===============================================================================
#    Copyright 2017 Geoscience Australia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ===============================================================================

'''
Created on 21/10/202
@author: Neil Symington
Updates workflow for converting AEM conductivity models to segy (from https://github.com/Neil-Symington/AEM2SEG-Y).
This approach utilised netcdf formatted conductivity rather than aseg-gdf files. This makes it easier to implement
as the variables can be called by their name rather than by referring to column numbers.
This relies on a python environment with obspy0.10 .  See this document for installation
https://docs.google.com/document/d/1Elv19V3QclRzz9MhVqGacGvdHT4xERTr6etYUTFTpcA
The netcdf format for line geophysical data is from geophys_utils
https://github.com/GeoscienceAustralia/geophys_utils
'''

import sys, os, glob
import math
import numpy as np
import aem2segy
# Needs to be obspy 0.10
import obspy
import netCDF4

# Path to netcdf file
infile = r"C:\Users\symin\OneDrive\Documents\GA\AEM\rjmcmc\Injune_petrel_rjmcmc_pmaps.nc"

# initiate netcdf dataset
d = netCDF4.Dataset(infile)

# we will use all of the lines.
lines = d.variables['line'][:]

# Define the key variables

yres = 2. # vertical resolution
conductivity_var = 'conductivity_p50'
max_depth = 350.
# datum is maximum elevation rounded to the nearest 10
datum = np.float(math.ceil(np.max(d.variables['elevation'][:])/10.)) * 10.
outdir = r"C:\temp\Injune_segy"

# Create the output directory
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Define the new elevation profile onto which the conductivity data will be interpolated
z_new = np.arange(-1 * datum + (yres / 2.), max_depth + (yres / 2.), yres)

for line in lines:

    # Now we extract our data
    data_dict = aem2segy.get_sorted_line_data(d, line, vars = ['fiducial', conductivity_var, 'easting',
                                                               'northing', 'elevation'], sort_on = 'fiducial')

    data_dict['layer_top_depth'] = d.variables['layer_centre_depth'] - d.variables['layer_centre_depth'][0]
    # Define numpy array for continuous, interpolated conductivity data

    stream_cond = np.ones((np.shape(data_dict['fiducial'])[0], math.ceil((datum + max_depth) / yres)))
    # Iterate through each fiducial, interpolate and write the results into the corresponding row in the array
    for i in range(len(stream_cond)):
        interp_dat = aem2segy.interpolate_layer_data(data_dict['layer_top_depth'], z_new,
                                                     data_dict[conductivity_var][i], data_dict['elevation'][i],
                                                     max_depth, datum)
        stream_cond[i] = 10**interp_dat

    print(stream_cond.shape)

    # Now write each fiducial into the trace. For this we use the obspy package
    traces = []

    for row in stream_cond:
        trace = obspy.core.trace.Trace(row)
        # This is a hack to try and produce a segy with an acceptably low
        # sampling frequency. I highly recommend to include sample depth information
        # in the header
        trace.stats.delta = yres * 0.001
        traces.append(trace)
    # Write the traces into a stream
    stream = obspy.core.stream.Stream(traces)

    for tr in stream:
        tr.data = np.require(tr.data, dtype=np.float32)

    # Write the stream into a temporary segy
    tempoutfile = os.path.join(outdir, str(line) + '_temp.segy')

    stream.write(tempoutfile, format='SEGY', data_encoding=1)

    # Reopen the segy

    st = obspy.segy.core.readSEGY(tempoutfile)

    os.remove(tempoutfile)

    # Now write in some of the important header information

    st.stats.binary_file_header['line_number'] = int(line)

    # Now write trace header information

    for i in range(len(st)):
        st[i].stats.segy.trace_header.trace_sequence_number_within_line = i
        st[i].stats.segy.trace_header.trace_number_within_the_original_field_record = data_dict['fiducial'][i]
        st[i].stats.segy.trace_header.trace_number_within_the_ensemble = data_dict['fiducial'][i]
        st[i].stats.segy.trace_header.ensemble_number = data_dict['fiducial'][i]
        st[i].stats.segy.trace_header.source_coordinate_x = data_dict['easting'][i]
        st[i].stats.segy.trace_header.source_coordinate_y = data_dict['northing'][i]
        st[i].stats.segy.trace_header.group_coordinate_x = data_dict['easting'][i]
        st[i].stats.segy.trace_header.group_coordinate_y = data_dict['northing'][i]
        st[i].stats.segy.trace_header.receiver_group_elevation = data_dict['elevation'][i]
        st[i].stats.segy.trace_header.datum_elevation_at_receiver_group = 0
        st[i].stats.segy.trace_header.datum_elevation_at_source = 0
        st[i].stats.segy.trace_header.delay_recording_time = -1 * datum

    outfile = os.path.join(outdir, str(line) + '.segy')

    st.write(outfile, format='SEGY', data_encoding=1)