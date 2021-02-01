# Import python modules
import numpy as np
import yaml
import netCDF4
import os
from dash.exceptions import PreventUpdate
from garjmcmctdem_utils import spatial_functions, aem_utils, netcdf_utils
from garjmcmctdem_utils import plotting_functions as plots
from garjmcmctdem_utils.misc_utils import pickle2xarray, xarray2pickle
import warnings
import pandas as pd
import geopandas as gpd
from shapely import wkt
warnings.filterwarnings('ignore')
# Dash dependencies
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
import matplotlib
import base64, io
import gc
import math


#yaml_file = "dash_workflow/interpretation_config.yaml"
yaml_file = "interpretation_config.yaml"
settings = yaml.safe_load(open(yaml_file))

interp_settings, model_settings, AEM_settings, det_inv_settings, stochastic_inv_settings, section_settings,\
borehole_settings, crs = settings.values()

# Set key variables for convenience
uncertainty_settings = stochastic_inv_settings['uncertainty']

root = interp_settings['data_directory']

lines = interp_settings['lines']

# Prepare AEM data
em = aem_utils.AEM_data(name = AEM_settings['name'],
                        system_name = AEM_settings['system_name'],
                        netcdf_dataset = netCDF4.Dataset(os.path.join(root, AEM_settings['nc_path'])))

# Grid the data if the user wants
if AEM_settings["grid_sections"]:
    print("Gridding AEM data. This may take a few minutes.")
    ## TODO add path checking function
    outdir = os.path.join(root, AEM_settings['section_directory'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    grid_vars = AEM_settings['grid_variables']
    em.griddify_variables(variables=grid_vars, lines=lines, save_to_disk=True, output_dir = outdir)

# Prepare deterministic inversion

det = aem_utils.AEM_inversion(name = det_inv_settings['inversion_name'],
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, det_inv_settings['nc_path'])))

if det_inv_settings["grid_sections"]:
    print("Gridding deterministic AEM inversion. This may take a few minutes.")
    ## TODO add path checking function
    outdir = os.path.join(root, det_inv_settings['section_directory'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    det.grid_sections(variables = det_inv_settings['grid_variables'], lines = lines,
                      xres = det_inv_settings['horizontal_resolution'],
                      yres = det_inv_settings['verticaL_resolution'],
                      return_interpolated = False, save_to_disk = True,
                      output_dir = outdir)
else:
    pass

if det_inv_settings['plot_grid']:
    # loaad layer grids
    grid_file = os.path.join(root, det_inv_settings['layer_grid_path'])
    # Not enough density to plot grid
    det.load_lci_layer_grids_from_pickle(grid_file)

# Create polylines
det.create_flightline_polylines(crs = crs['projected'])

gdf_lines = det.flightlines[np.isin(det.flightlines['lineNumber'], lines)]

# Prepare stochastic inversion

rj = aem_utils.AEM_inversion(name = stochastic_inv_settings['inversion_name'],
                              inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, stochastic_inv_settings['nc_path'])))


if stochastic_inv_settings["grid_sections"]:
    print("Gridding stochastic AEM inversion. This may take a few minutes.")
    outdir = os.path.join(root, stochastic_inv_settings['section_directory'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    rj.grid_sections(variables = stochastic_inv_settings['grid_variables'], lines = lines,
                      xres = stochastic_inv_settings['horizontal_resolution'],
                      yres = stochastic_inv_settings['verticaL_resolution'],
                      return_interpolated = False, save_to_disk = True,
                      output_dir = outdir)
else:
    pass

# Prepare borehole data
if borehole_settings['include']:
    infile = os.path.join(root, borehole_settings['borehole_file'])
    ## TODO add a chacking function
    cols = ['WELL', 'TOP_AHD_M', 'BASE_AHD_M', 'GA_UNIT', 'Strat_name',
            'TOP_MD_M', 'BASE_MD_M', 'fiducial', 'line', 'geometry']
    df_bh = pd.read_csv(infile)[cols]
    geom = [wkt.loads(s) for s in df_bh['geometry']]
    df_bh['easting'] = [coord.x for coord in geom]
    df_bh['northing'] = [coord.y for coord in geom]

    # We need to get these data into the same reference frame as our grids
    df_bh['distance_along_line'] = np.nan
    df_bh['AEM_elevation'] = np.nan

# To reduce the amount of data that is stored in memory the section data are stored as xarrays in pickle files
#  We will only bring them into memory as needed. Here we point the inversions to their pickle files

em.section_path = {}
det.section_path = {}
rj.section_path = {}
rj.distance_along_line = {}

# Iterate through the lines
for lin in lines:
    # Add path as attribute
    em.section_path[lin] = os.path.join(root, AEM_settings['section_directory'],
                                        "{}.pkl".format(str(lin)))
    det.section_path[lin] = os.path.join(root, det_inv_settings['section_directory'],
                                         "{}.pkl".format(str(lin)))
    rj.section_path[lin] = os.path.join(root, stochastic_inv_settings['section_directory'],
                                        "{}.pkl".format(str(lin)))
    # Using this gridding we find the distance along the line for each garjmcmc site
    # Get a line mask
    line_mask = netcdf_utils.get_lookup_mask(lin, rj.data)
    # get the coordinates
    line_coords = rj.coords[line_mask]

    em_section_data = pickle2xarray(em.section_path[lin])

    dists = spatial_functions.xy_2_var(em_section_data,
                                      line_coords,
                                      'grid_distances')

    # Add a dictionary with the point index distance along the line to our inversion instance
    rj.distance_along_line[lin] = pd.DataFrame(data = {"point_index": np.where(line_mask)[0],
                                                       "distance_along_line": dists,
                                                       'fiducial': rj.data['fiducial'][line_mask]}
                                               ).set_index('point_index')
    # If we are gridding the stochastic inversions, then we scale them to the deterministic inversions
    if stochastic_inv_settings["grid_sections"]:

        rj_section_data = pickle2xarray(rj.section_path[lin])

        try:

            rj_section_data['grid_distances'] = spatial_functions.scale_distance_along_line(em_section_data,
                                                                                            rj_section_data)
        except IndexError: # This happens in cases where there are insufficient points on a line for interpolation
            pass

        # Save xarray back to pickle file

        xarray2pickle(rj_section_data, rj.section_path[lin])

    #Calculate distance along the line for the boreholes
    if borehole_settings['include']:
        line_mask = df_bh['line'] == lin
        df_bh_ss = df_bh[line_mask]
        if len(df_bh_ss) > 0:
            bh_coords = df_bh_ss[['easting', 'northing']].values
            dists_ = spatial_functions.xy_2_var(em_section_data,
                                               bh_coords,
                                               'grid_distances',
                                                max_distance = 500.)
            df_bh.at[df_bh_ss.index, 'distance_along_line'] = dists_
            elevs_ = spatial_functions.xy_2_var(em_section_data,
                                               bh_coords,
                                               'elevation',
                                                max_distance = 500.)
            df_bh.at[df_bh_ss.index, 'AEM_elevation'] = elevs_
    # Remove from memory
    rj_section_data = None
    det_section_data = None
    gc.collect()


# Define colour stretch for em data

viridis = cm.get_cmap('viridis')
lm_colours = [matplotlib.colors.rgb2hex(x) for x in viridis(np.linspace(0, 1, 18))]

plasma = cm.get_cmap('plasma')
hm_colours = [matplotlib.colors.rgb2hex(x) for x in plasma(np.linspace(0, 1, 23))]

def list2options(list):
    options = []
    for l in list:
        options.append({'label': str(l), 'value': l})
    return options

def subset_df_by_line(df_, line, line_col = 'SURVEY_LINE'):
    mask = df_[line_col] == line
    return df_[mask]

def style_from_surface(surfaceNames, df_template,  styleName):
    style = []


    for item in surfaceNames:
        prop = df_template[df_template['SurfaceName'] == item][styleName]
        style.append(prop)
    return style

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')), keep_default_na=False, dtype = dtypes)
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded), dtype = dtypes)

# section functions
def xy2fid(x,y, dataset):
    dist, ind = spatial_functions.nearest_neighbours([x, y],
                                                     dataset.coords,
                                                     max_distance = 100.)
    return dataset.data['fiducial'][ind][0]

def find_trigger():
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    return trig_id

def interp2scatter(line, xarr, interpreted_points, easting_col = 'X',
                   northing_col = 'Y', elevation_col = 'ELEVATION',
                   line_col = 'SURVEY_LINE'):

    utm_coords = np.column_stack((xarr['easting'].values, xarr['northing'].values))

    df_ = subset_df_by_line(interpreted_points, line, line_col = line_col)

    dist, inds = spatial_functions.nearest_neighbours(df_[[easting_col,northing_col]].values,
                                                      utm_coords, max_distance=100.)

    grid_dists = xarr['grid_distances'].values[inds]
    elevs = df_[elevation_col].values

    return grid_dists, elevs

def dash_conductivity_section(section, line, vmin, vmax, cmap, xarr, pmap_kwargs):

    # Create subplots
    fig = make_subplots(rows=4, cols = 1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.2, 0.05, 0.70, 0.05])
    vmin = np.log10(vmin)
    vmax = np.log10(vmax)
    logplot = True

    # Extract data based on the section plot keyword argument
    if section == 'lci':
        misfit = xarr['data_residual'].values
        z = np.log10(xarr['conductivity'].values)

    elif section.startswith('rj'):
        misfit = xarr['misfit_lowest'].values
        if section == "rj-p50":
            z = np.log10(xarr['conductivity_p50'])
        elif section == "rj-p10":
            z = np.log10(xarr['conductivity_p10'])
        elif section == "rj-p90":
            z = np.log10(xarr['conductivity_p90'])
        elif section == "rj-lpp":
            z = xarr['interface_depth_histogram'] / rj.data.histogram_samples
            vmin = 0.01
            vmax = 0.8
            cmap = 'greys'
            logplot = False

    tickvals = np.linspace(vmin, vmax, 5)
    if logplot:
        ticktext = [str(np.round(x, 3)) for x in 10**tickvals]
    else:
        ticktext = [str(np.round(x, 3)) for x in tickvals]

    dist_along_line = xarr['grid_distances'].values
    grid_elevations = xarr['grid_elevations'].values
    elevation = xarr['elevation'].values
    easting = xarr['easting'].values
    northing = xarr['northing'].values

    # plot the data residual
    fig.add_trace(go.Scatter(x = dist_along_line,
                             y = misfit,
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = None,
                             name = "residual"),

                       )


    fig.add_trace(go.Heatmap(z = z,
                            zmin = vmin,
                            zmax = vmax,
                            x = dist_along_line,
                            y = grid_elevations,
                            colorscale =cmap,
                            colorbar=dict(
                                 title="conductivity",
                                 tickvals=tickvals,
                                 ticktext=ticktext
                             ),
                            hoverinfo="none",
                             name="section"
                            ),
                      row = 3, col = 1
        )


    # Add the elevation
    fig.add_trace(go.Scatter(x = dist_along_line,
                             y = elevation,
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = 'skip'),
                  row = 3, col = 1,)

    # Create a list of easting and northing (strings) for label
    labels = []
    for i in range(len(easting)):
        labels.append(' '.join(["x:", "{:.1f}".format(easting[i]), "y:", "{:.1f}".format(northing[i])]))
    # Add the easting/ northing as a scatter plot
    fig.add_trace(go.Scatter(x=dist_along_line,
                             y=northing,
                             line=dict(color='black', width=3),
                             hovertext = labels,
                             showlegend=False,
                             name = "coordinates"),
                  row=4, col=1, )

    df_rj_sites = rj.distance_along_line[line]

    labels = ["netcdf_point_index = " + str(x) for x in df_rj_sites.index]
    colours = len(df_rj_sites) * ['purple']
    symbols = len(df_rj_sites) * ['circle']
    marker_size = len(df_rj_sites) * [5]
    if len(pmap_kwargs) > 0:
        pt_idx = pmap_kwargs['point_idx']
        try:
            idx = np.where(df_rj_sites.index == pt_idx)[0][0]
            colours[idx] = 'red'
            symbols[idx] = 'triangle-up'
            marker_size[idx] = 10.
        except IndexError:
            pass
    site_distances = df_rj_sites['distance_along_line'].values
    #site_plot_elevation = 20. + np.max(xarr['elevation'].values) * np.ones(shape=len(df_rj_sites), dtype=np.float)
    fig.add_trace(go.Scatter(x=site_distances,
                             y=np.zeros(shape = site_distances.shape),
                             mode='markers',
                             marker_symbol = symbols,
                             marker_color = colours,
                             hovertext=labels,
                             marker_size = marker_size,
                             showlegend=False),
                  row=2, col=1)
    # Reverse y-axis
    fig.update_yaxes(autorange=True, row = 1, col = 1, title_text = "data residual")
    fig.update_yaxes(autorange=True, row=3, col=1, title_text="elevation (mAHD)")
    fig.update_yaxes(visible= False, showticklabels= False, row=2,col=1)
    fig.update_yaxes(visible=False, showticklabels=False, row=4, col=1)

    fig.update_xaxes(title_text= "distance along line " + " (m)", row=3, col=1)
    fig['layout'].update({'height': 600})
    return fig

def dash_EM_section(line):

    # Create subplots
    fig = make_subplots(rows=4, cols = 1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.1, 0.1, 0.4, 0.4])
    # get data
    xarr = pickle2xarray(em.section_path[line])
    grid_distances = xarr['grid_distances'].values
    # Subset based on line length to avoid long rendering for plots
    ss = np.int(math.ceil(np.max(grid_distances)/10000.))
    grid_distances = grid_distances[::ss]
    plni = xarr['powerline_noise'].values[::ss]
    tx_height = xarr['tx_height_measured'].values[::ss]
    lm_data = xarr['low_moment_Z-component_EM_data'].values[::ss]
    hm_data = xarr['high_moment_Z-component_EM_data'].values[::ss]

    fig.add_trace(go.Scatter(x=grid_distances,
                             y=tx_height,
                             line=dict(color='black', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=1, col=1, )
    fig.add_trace(go.Scatter(x=grid_distances,
                             y=plni,
                             line=dict(color='black', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=2, col=1, )


    # Adaptive marker size
    #size = 1200./lm_data.shape[0]

    # prepare the line data
    for j in range(lm_data.shape[1]):
        labels = ["low moment gate " + str(j)]
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=lm_data[:,j],
                                 mode='lines',
                                 line={
                                     "color": lm_colours[j]
                                 },
                                 hovertext=labels,
                                 showlegend=False),
                      row=3, col=1, )

    # prepare the line data
    for j in range(hm_data.shape[1]):
        labels = ["low moment gate " + str(j)]
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=hm_data[:, j],
                                 mode='lines',
                                 line={
                                     "color": hm_colours[j]
                                 },
                                 hovertext=labels,
                                 showlegend=False),
                      row=4, col=1, )


    fig.update_yaxes(title_text="tx height (m)", row=1, col=1)
    fig.update_yaxes(title_text="PLNI", type="log", row=2, col=1)
    fig.update_yaxes(title_text="LMZ data (pV/Am^4)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="HMZ data (pV/Am^4)", type="log", row=4, col=1)
    fig.update_xaxes(title_text="distance along line " + " (m)", row=4, col=1)
    fig['layout'].update({'height': 600})

    return fig

def plot_section_points(fig, line, df_interp, xarr, select_mask):

    # Create a scatter plot on the section using projection
    interpx, interpz = interp2scatter(line, xarr, df_interp)

    if len(interpx) > 0:
        #labels = ["surface = " + str(x) for x in df_interp['BoundaryNm'].values]

        colours = df_interp["Colour"].values
        markers = df_interp["Marker"].values
        markerSize = df_interp["MarkerSize"].values
        # To make selected points obvious
        if len(select_mask) > 0:
            for idx in select_mask:
                markerSize[idx] += 10.

        fig.add_trace(go.Scatter(x = interpx,
                        y = interpz,
                        mode = 'markers',
                        hoverinfo = 'skip',#labels,
                        marker = {"symbol": markers,
                                  "color": colours,
                                  "size": markerSize
                                  },
                        name = 'interpretation',
                        showlegend = True,
                                 xaxis= 'x3',
                                 yaxis='y3')
                      )
        fig.update_layout()
    return fig

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

    # Need to create this for opening pickle files with xarrays
    xarr = pickle2xarray(det.section_path[line])

    dist = spatial_functions.xy_2_var(xarr, np.array([[easting, northing]]), 'grid_distances')

    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'depth_cells': depth_cells,
            'cond_cells': cond_cells, 'det_cond': det_cond, 'det_depth_top': det_depth_top, 'det_doi': det_doi,
            'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial,
            'elevation': elevation, 'det_dist': dist, 'det_line': xarr, 'n_histogram_samples': n_hist_samples}


def dash_pmap_plot(point_index):
    # Extract the data from the netcdf data
    D = extract_rj_sounding(rj, det, point_index)
    pmap = D['conductivity_pdf']
    x1,x2,y1,y2 = D['conductivity_extent']
    n_depth_cells, n_cond_cells  = pmap.shape

    x = np.linspace(x1,x2, n_cond_cells)
    y = D['elevation'] - np.linspace(y2,y1, n_depth_cells)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        horizontal_spacing=0.01,
                        column_widths=[0.7, 0.3])

    fig.add_trace(go.Heatmap(z = pmap,
                            zmin = 0,
                            zmax = np.max(pmap),
                            x = x,
                            y = y,
                            colorscale ='plasma',
                            colorbar=dict(
                                 title="probability",
                             )),
                             row = 1, col = 1)

    #  PLot the median, and percentile plots
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p10']),
                             y = y,#D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'black',
                                     "width": 1.},
                             name = "p10 conductivity",
                             showlegend = False),
                  row = 1, col=1)
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p90']),
                             y = y,#D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'black',
                                     "width": 1.},
                             name = "p90 conductivity",
                             showlegend = False),
                  row = 1, col=1)
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p50']),
                             y = y,#D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'gray',
                                     "width": 1.,
                                     'dash': 'dash'},
                             name = "p50 conductivity",
                             showlegend = False),
                  row = 1, col=1)

    det_expanded, depth_expanded = plots.profile2layer_plot(D['det_cond'], D['det_depth_top'])

    fig.add_trace(go.Scatter(x=np.log10(det_expanded),
                             y= D['elevation'] - depth_expanded,
                             mode='lines',
                             line={"color": 'pink',
                                   "width": 1.,
                                   'dash': 'dash'},
                             name=det.name,
                             showlegend=False),
                  row = 1, col=1)

    fig.add_trace(go.Scatter(x=D['change_point_pdf'],
                             y= y,
                             mode='lines',
                             line={"color": 'black',
                                   "width": 2.},
                             name=det.name,
                             showlegend=False),
                  row = 1, col=2)

    fig.update_layout(
        autosize=False,
        height=600)

    fig.update_layout(xaxis=dict(scaleanchor = 'y',
                                 scaleratio = 100.))
    return fig

def flightline_map(line, vmin, vmax, layer):

    fig = go.Figure()

    cond_grid = np.log10(det.layer_grids['Layer_{}'.format(layer)]['conductivity'])

    x1, x2, y1, y2 = det.layer_grids['bounds']
    n_y_cells, n_x_cells = cond_grid.shape
    x = np.linspace(x1, x2, n_x_cells)
    y = np.linspace(y2, y1, n_y_cells)

    fig.add_trace(go.Heatmap(z=cond_grid,
                               zmin=np.log10(vmin),
                               zmax=np.log10(vmax),
                               x=x,
                               y=y,
                              colorscale="jet"
                             ))

    for linestring, lineNo in zip(gdf_lines.geometry, gdf_lines.lineNumber):

        if int(lineNo) == int(line):
            c = 'red'
        else:
            c = 'black'
        x, y = linestring.xy

        fig.add_trace(go.Scatter(x = list(x),
                                 y = list(y),
                                 mode = 'lines',
                                 line = {"color": c,
                                         "width": 2.},
                                 name = str(lineNo)))

    xmin, xmax = np.min(rj.data['easting'][:]) - 5000., np.max(rj.data['easting'][:]) + 5000.
    ymin, ymax = np.min(rj.data['northing'][:]) - 5000., np.max(rj.data['northing'][:]) + 5000.

    fig.update_layout(yaxis=dict(range=[ymin, ymax],
                                 scaleanchor = 'x',
                                 scaleratio = 1.),
                      xaxis=dict(range=[xmin, xmax]))
    # PLot the boreholes
    if borehole_settings['include']:
        grouped = df_bh.groupby('WELL').mean()
        coords = grouped[['easting', 'northing']].values
        wells = grouped.index.values
        fig.add_trace(go.Scatter(x = coords[:,0],
                                   y=coords[:,1],
                                  mode = 'markers',
                                   marker={"color": 'blue',
                                         "size": 4.
                                         },
                                   hovertext=wells))
    #fig['data'][0]['showscale'] = False

    return fig

#Define key functions
def full_width_half_max(interpreted_depth, depth_array, count_array):
    """
    Function for calculating the interval that is >0.5 times the probability
    Parameters
    ----------
    interpreted_depth: float of the interpreted depth
    depth_array: array of depths
    count_array: array of counts. nb this is a proxy for probabilities

    Returns
    -------
    tuple of indices denoting the top and bottom of the full width half max
    """

    idx_upper = -1
    idx_lower = -1

    # Find the maximum depth index
    max_idx = np.argmin(np.abs(depth_array - interpreted_depth))

    # get the maximum probability
    fmax = count_array[max_idx]

    # Snap to maximum likelihood depth
    if uncertainty_settings['snap_interpretation']:
        # find the search window
        window = int(np.ceil((uncertainty_settings['snap_window']/2.)/2))

        window_slice = [max_idx - window, max_idx + window]
        # to ensure no index errors
        if window_slice[0] < 0:
            window_slice[0] = 0
        if window_slice[1] >= len(count_array):
            window_slice[1] = len(count_array)

    # positive direction

    for idx in np.arange(max_idx, depth_array.shape[0]):
        if count_array[idx] <= fmax/2.:
            idx_upper = idx
            break
    # negative direction
    for idx in np.arange(max_idx, -1, -1):
        if count_array[idx] <= fmax/2.:
            idx_lower = idx
            break
    # Now calculate the width
    return (idx_lower, max_idx, idx_upper)

def uncertainty_from_pmap(point_idx, interpreted_depth):
    """
    Fucntion for estimating uncertainty from pmap and depth estimate
    Parameters
    ----------
    point_idx: int
        rj netcdf point index
    interpreted_depth: float
        depth of user interpretation
    Returns
    -------
    Uncertainty: float
    """


    depth_array = rj.data['layer_centre_depth'][:]
    count_array = rj.data['interface_depth_histogram'][point_idx]
    [lower_ind, max_ind, upper_ind] = full_width_half_max(interpreted_depth, depth_array, count_array)

    if np.logical_and(lower_ind != -1, upper_ind != -1):
        if uncertainty_settings['snap_interpretation']:
            # Snap to maximum depth
            interpreted_depth = depth_array[max_ind]
        uncertainty = depth_array[upper_ind] - depth_array[lower_ind]
        return interpreted_depth, uncertainty
    else:
        return interpreted_depth, np.nan

def plot_borehole_segments(fig, df):
    # Plot each row at a time
    for index, row in df.iterrows():
        y = row[['AEM_elevation', 'AEM_elevation']].values - row[['TOP_MD_M', 'BASE_MD_M']].values
        # For visualisation we will give units with no lower depth a thickness of 4m
        if np.isnan(y[1]):
            y[1] = y[0] - 4.
        x = row[['distance_along_line', 'distance_along_line']].values
        colour = borehole_settings['unit_colours'][row['Strat_name']]
        labels = [row['GA_UNIT']] * 2
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines+markers',
                                 line={
                                     "color": colour,
                                     'width': 3.,
                                 },
                                 marker_size = 3.,
                                 marker_symbol = 'hash',
                                 hovertext=labels,
                                 name = 'boreholes',
                                 showlegend=False),
                      row=3, col=1, )
    return fig

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

# To do add some check in here
df_model_template = pd.read_csv(model_settings['templateFile'])

# get the data types from the template file and add the interpretation specific datatypes
dtypes = df_model_template.dtypes.to_dict()

dtypes['OvrStrtCod'] = 'int32'
dtypes['UndStrtCod'] = 'int32'
dtypes["MarkerSize"] = 'int8'
dtypes['point_index'] = 'int16'
dtypes['fiducial'] = 'float64'
dtypes['X'] = 'float64'
dtypes['Y'] = 'float64'
dtypes['ELEVATION'] = 'float32'
dtypes['DEM'] = 'float32'
dtypes['DEPTH'] = 'float32'
dtypes['UNCERTAINTY'] = 'float32'
dtypes['SURVEY_LINE'] = 'int32'

df_interpreted_points = pd.read_csv(model_settings['interpFile'],
                                    dtype = dtypes)

# we are storing the preferred plotting properties for each point to reduce how often we need to do lookups of the template df
colour, marker, marker_size = [], [], []

for index, row in df_interpreted_points.iterrows():
    surfName = row['BoundaryNm']
    mask = df_model_template['SurfaceName'] == surfName
    colour.append(df_model_template[mask]['Colour'].values[0])
    marker.append(df_model_template[mask]['Marker'].values[0])
    marker_size.append(df_model_template[mask]['MarkerSize'].values[0])

df_interpreted_points['Colour'] = colour
df_interpreted_points['Marker'] = marker
df_interpreted_points['MarkerSize']  = marker_size

# for use with the dropdown
line_options = list2options(lines)

surface_options = list2options(df_model_template['SurfaceName'].values)

app.layout = html.Div([
    html.Div(
                [
                    html.Div(html.H1(' '.join([str(model_settings['name']), " AEM interpretation app"])),
                             className= "three columns"),
                    html.Div([html.H4("Select surface"),
                              dcc.Upload(
                                        id='template-upload',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select'),
                                            ' a surface template file'
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                # Allow multiple files to be uploaded
                                multiple=False),
                             dcc.Dropdown(id = "surface_dropdown",
                                            options=surface_options,
                                            value=(surface_options[0]['label']))

                             ],className = "three columns"),
                    html.Div([html.H4("Select section"),
                             dcc.Dropdown(id = "section_dropdown",
                                            options=[
                                                    {'label': 'laterally constrained inversion',
                                                     'value': 'lci'},
                                                    {'label': 'garjmcmctdem_utils - p50',
                                                     'value': 'rj-p50'},
                                                    {'label': 'garjmcmctdem_utils - p10',
                                                     'value': 'rj-p10'},
                                                    {'label': 'garjmcmctdem_utils - p90',
                                                     'value': 'rj-p90'},
                                                    {'label': 'garjmcmctdem_utils - layer probability',
                                                     'value': 'rj-lpp'}],
                                            value="lci"),

                             ],className = "three columns"),
                    html.Div([html.H4("Select line"),
                             dcc.Dropdown(id = "line_dropdown",
                                            options=line_options,
                                            value= line_options[0]['value']),
                             ],className = "three columns")
                ], className = 'row'
            ),
    html.Div(
            [
                html.Div(html.Div(id='message'),
                         className = "three columns"),
                html.Div(dash_table.DataTable(id='surface_table',
                                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                              columns = [{"name": i, "id": i} for i in df_model_template.columns],
                                              data=df_model_template.to_dict('records'),
                                                editable=True,
                                            fixed_columns={ 'headers': True},
                                            sort_action="native",
                                            sort_mode="multi",
                                            row_selectable=False,
                                            row_deletable=False,
                                              style_table={
                                                  'maxHeight': '100px',
                                                  'overflowY': 'scroll',
                                                  'maxWidth': '500px',
                                                  'overflowX': 'scroll'}
                                              ),
                         className = "three columns"),
                html.Div([html.Div(["Conductivity plotting minimum: ", dcc.Input(
                                    id="vmin", type="number",
                                    min=0.001, max=10, value = section_settings['vmin'])],
                         className = 'row'),
                         html.Div(["Conductivity plotting maximum: ", dcc.Input(
                                    id="vmax", type="number",
                                    min=0.001, max=10, value = section_settings['vmax'])],
                         className='row'),
                         html.Div(["AEM layer grid: ", dcc.Input(
                                    id="layerGrid", type="number",
                                    min=1, max=30, value = 1, step = 1)],
                         className='row'),
                    ],
                    className = "three columns"),
                html.Div([

                         html.Div(html.Button('Export results', id='export', n_clicks=0),
                                  className = 'row'),
                         html.Div(dcc.Input(id='export-path', type='text', placeholder = 'Input valid output path'),
                                 className= 'row'),
                         html.Div(id='export_message', className= 'row')
                          ],
                         className= "three columns"),
             ], className = "row"),
    html.Div([
            dcc.Tabs(id='section_tabs', value='conductivity_section', children=[
                    dcc.Tab(label='Conductivity section', value='conductivity_section'),
                    dcc.Tab(label='AEM data section', value='data_section'),
                 ]),
        html.Div([dcc.Graph(
            id='section_plot',
        )], style={'height': '600'}),
        ],style = {'marginTop': 20}),
    html.Div([html.Div(
        html.Div([
            dash_table.DataTable(id='interp_table',
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                columns = [{"name": i, "id": i} for i in df_interpreted_points.columns],
                                #data=model.interpreted_points[model.interpreted_points['SURVEY_LINE'] == int(model.line_options[0]['label'])].to_dict('records'),
                                fixed_columns={ 'headers': True},
                                sort_action="native",
                                sort_mode="multi",
                                row_selectable="multi",
                                row_deletable=True,
                                selected_columns=[],
                                selected_rows=[],
                                style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                           'height': '40px'},
                                style_cell={
                                             'backgroundColor': 'rgb(50, 50, 50)',
                                             'color': 'white',
                                             'minHeight': '50px',
                                             'minWidth': '0px', 'maxWidth': '800px',
                                             'whiteSpace': 'normal',
                                             'font-size': '12px'
                                         },
                              style_table={
                                          'maxHeight': '400px',
                                          'overflowY': 'scroll',
                                          'maxWidth':  '90%',
                                          'overflowX': 'scroll'}),

        ],
            className = 'row')
        , className = "six columns"),
        html.Div([
            dcc.Tabs(id='map_tabs', value='map_plot', children=[
                dcc.Tab(label='Map plot', value='map_plot'),
                dcc.Tab(label='Pmap plot', value='pmap_plot'),
            ]),
            html.Div(id='map-tabs-content', style = {'height': '600px'})
        ],
            className = "six columns")],
        className = 'row'),
    html.Div(id = 'output', style={'display': 'none'}),
    dcc.Store(id='interp_memory'),
    dcc.Store(id = "model_memory"), # storage for model template file
    dcc.Store(id = 'pmap_store')

])

# megacallback function for updating the interpreted points. This is either done by clicking on the section or deleting
# from the table
@app.callback(
    [Output('interp_memory', 'data'),
     Output('section_plot', 'figure'),
     Output('interp_table', 'data'),
     Output('pmap_store', 'data')],
    [Input('section_plot', 'clickData'),
     Input('interp_table', 'data_previous'),
     Input('section_dropdown', 'value'),
     Input('section_tabs', 'value'),
     Input("line_dropdown", 'value'),
     Input('vmin', 'value'),
     Input('vmax', 'value'),
     Input('interp_table', "derived_virtual_selected_rows"),],
     [State('interp_table', 'data'),
      State("surface_dropdown", 'value'),
      State("interp_memory", 'data'),
      State("model_memory", "data"),
      State('pmap_store', 'data')])
def update_many(clickData, previous_table, section, section_tab, line, vmin, vmax, selected_rows, current_table,
                        surfaceName, interpreted_points, model, pmap_store):
    trig_id = find_trigger()

    #a bunch of "do nothing" cases here - before doing any data transformations to save
    #on useless computation
    if trig_id == 'section_plot.clickData' and section_tab == 'data_section':
        #clicked on the data section so nothing to do
        raise PreventUpdate

    # Access the data from the store. The or is in case of None callback at initialisation
    interpreted_points = interpreted_points or df_interpreted_points.to_dict('records')

    if selected_rows is None:
        selected_rows = []

    # Guard against empty lists
    if len(interpreted_points) == 0:
        df = df_interpreted_points.copy()
    else:
        df = pd.DataFrame(interpreted_points).infer_objects()

    pmap_store = pmap_store or {}

    if section == "lci":
        xarr = pickle2xarray(det.section_path[line])
    else:
        xarr = pickle2xarray(rj.section_path[line])

    if trig_id == 'section_plot.clickData' and section_tab == 'conductivity_section':
        if clickData['points'][0]['curveNumber'] == 1:
            # Get the interpretation data from the click function
            model = model or df_model_template.to_dict('records')
            df_model = pd.DataFrame(model).infer_objects()
            row = df_model[df_model['SurfaceName'] == surfaceName].squeeze()

            eventxdata, eventydata = clickData['points'][0]['x'], clickData['points'][0]['y']
            min_idx = np.argmin(np.abs(xarr['grid_distances'].values - eventxdata))

            easting = xarr['easting'].values[min_idx]
            northing = xarr['northing'].values[min_idx]
            elevation = xarr['elevation'].values[min_idx]
            depth = elevation - eventydata

            # Estimate uncertainty
            if section == "lci":
                fid = xy2fid(easting, northing, det)
                uncertainty = np.nan
            else:
                # Find the nearest pmap
                distances = np.abs(rj.distance_along_line[line]['distance_along_line'] - eventxdata)

                pmap = rj.distance_along_line[line].iloc[np.argmin(distances)]
                fid = pmap['fiducial']
                pmap_index = rj.distance_along_line[line].index[np.argmin(distances)]

                if uncertainty_settings and np.min(distances) < uncertainty_settings['maximum_distance']:
                    # Calculate uncertainty
                    depth, uncertainty = uncertainty_from_pmap(pmap_index, depth)
                else:
                    uncertainty = np.nan


            # append to the surface object interpreted points
            interp = {'fiducial': fid,
                      'inversion_name': section,
                      'X': np.round(easting, 0),
                      'Y': np.round(northing, 0),
                      'DEPTH': np.round(depth, 0),
                      'ELEVATION': np.round(eventydata,1),
                      'DEM': np.round(elevation,1),
                      'UNCERTAINTY': uncertainty,
                      'Type': row.Type,
                      'BoundaryNm': row.SurfaceName,
                      'BoundConf': row.BoundConf,
                      'BasisOfInt': row.BasisOfInt,
                      'OvrConf': row.OvrConf,
                      'OvrStrtUnt': row.OvrStrtUnt,
                      'OvrStrtCod': row.OvrStrtCod,
                      'UndStrtUnt': row.UndStrtUnt,
                      'UndStrtCod': row.UndStrtCod,
                      'WithinType': row.WithinType,
                      'WithinStrt': row.WithinStrt,
                      'WithinStNo': row.WithinStNo,
                      'WithinConf': row.WithinConf,
                      'InterpRef': row.InterpRef,
                      'Comment': row.Comment,
                      'SURVEY_LINE': line,
                      'Operator': row.Operator,
                      "point_index": min_idx,
                      "Colour": row.Colour,
                      "Marker": row.Marker,
                      "MarkerSize": row.MarkerSize
                      }
            df_new = pd.DataFrame(interp, index=[0])
            df = pd.DataFrame(interpreted_points).append(df_new).infer_objects()
        elif clickData['points'][0]['curveNumber'] == 4:
            pmap_point_idx = int(clickData['points'][0]['hovertext'].split(" = ")[-1])
            pmap_store = {"point_idx": pmap_point_idx}

        else:
            raise PreventUpdate
    
    elif trig_id == 'interp_table.data_previous':

        if previous_table is None:
            raise PreventUpdate
        elif len(selected_rows) > 0:
            raise PreventUpdate

        else:
            # Compare dataframes to find which rows have been removed
            fids = []

            for row in previous_table:
                if row not in current_table:
                    fids.append(row['fiducial'])
                    # Remove from dataframe
            df = df[~df['fiducial'].isin(fids)]

    # Produce section plots
    df_ss = df[df['SURVEY_LINE'] == line]

    if section_tab == 'conductivity_section':
        fig = dash_conductivity_section(section, line,
                                        vmin=vmin,
                                        vmax=vmax,
                                        cmap=section_settings['cmap'],
                                        xarr = xarr,
                                        pmap_kwargs = pmap_store)


        # Check for selected rows which will be plotted differently
        if len(selected_rows) > 0:
            select_mask = np.where(np.isin(np.arange(0, len(df_ss)), selected_rows))[0]

        else:
            select_mask = []
        # PLot section
        if len(df_ss) > 0:
            fig = plot_section_points(fig, line, df_ss, xarr, select_mask=select_mask)
        # Now plot boreholes
        #df_bh_ss = df_bh[df_bh['line'] == line]
        #if len(df_bh_ss) > 0:
        #    # plot the boreholes as segments
        #    fig = plot_borehole_segments(fig, df_bh_ss)

        fig['layout'].update({'uirevision': line})

    elif section_tab == 'data_section':
        fig = dash_EM_section(line)

    return df.to_dict('records'), fig, df_ss.to_dict('records'), pmap_store


# Render map plot
@app.callback(Output('map-tabs-content', 'children'),
              [Input('map_tabs', 'value'),
               Input("line_dropdown", 'value'),
               Input('vmin', 'value'),
               Input('vmax', 'value'),
               Input('layerGrid', 'value'),
               Input('section_plot', 'clickData')])
def update_tab(tab, line, vmin, vmax, layer,
               clickData):
    if tab == 'map_plot':
        trig_id = find_trigger()
        #do nothing if active tab is the map and we only
        #clicked on the section
        if trig_id == 'section_plot.clickData':
            raise PreventUpdate
        fig = flightline_map(line, vmin, vmax, layer)
        return html.Div([
            dcc.Graph(
                id='polylines',
                figure=fig
            ),
        ])
    elif tab == 'pmap_plot':
        if clickData is not None:
            if clickData['points'][0]['curveNumber'] == 4:
                point_idx = int(clickData['points'][0]['hovertext'].split(" = ")[-1])

                fig = dash_pmap_plot(point_idx)
                return html.Div([
                    dcc.Graph(
                        id='pmap_plot',
                        figure=fig
                    ),
                ])
            else:
                raise PreventUpdate
        else:
            return html.Div(["Click on a purple point from the conductivity section to view the probability maps."])

@app.callback(
    Output('export_message', 'children'),
    [Input("export", 'n_clicks')],
    [State('export-path', 'value'),
    State('interp_memory', 'data')])
def export_data_table(nclicks, value, interpreted_points):
    interpreted_points = interpreted_points or df_interpreted_points.to_dict('records')
    df_interp = pd.DataFrame(interpreted_points).infer_objects()
    if np.logical_and(nclicks > 0, value is not None):
        if os.path.exists(os.path.dirname(value)):
            df_interp.reset_index(drop=True).to_csv(value, index=False)
            return "Successfully exported to " + value
        else:
            return value + " is an invalid file path."

@app.callback([Output("surface_dropdown", 'value'),
               Output("surface_dropdown", 'options'),
               Output("model_memory", "data"),
               Output("surface_table", "columns"),
               Output("surface_table", "data")],
              [Input('surface_table', 'data_timestamp'),
               Input('template-upload', 'contents')],
              [State('template-upload', 'filename'),
               State("interp_memory", 'data'),
               State("model_memory", "data"),
               State('surface_table', 'data')])
def update_surface(timestamp, contents, filename, interpreted_points, model_store, model_table):
    trig_id = find_trigger()

    model_store = model_store or df_model_template.to_dict("records")
    # Update the model store
    if trig_id == "surface_table.data_timestamp":
        df_model = pd.DataFrame(model_table).infer_objects()
    else:
        if contents is None:
            df_model = pd.DataFrame(model_store).infer_objects()
        else:
            df_model = parse_contents(contents, filename)
            valid = True#check_validity(df_model)
            ##TODO write a file validity function with a schema for eggs database
            if not valid:
                raise Exception("Input file is not valid")

    model_store = df_model.to_dict('records')
    return df_model['SurfaceName'][0], list2options(df_model['SurfaceName'].values), model_store, [{"name": i, "id": i} for i in df_model.columns], model_store

app.run_server(debug = False)
