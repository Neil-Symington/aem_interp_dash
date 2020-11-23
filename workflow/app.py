# Import python modules
import numpy as np
import yaml
import netCDF4
import sys, os
sys.path.append("../scripts")
from dash.exceptions import PreventUpdate
import spatial_functions
import aem_utils
import netcdf_utils
import modelling_utils
import plotting_functions as plots
from misc_utils import pickle2xarray, xarray2pickle
import warnings
import pandas as pd
import pickle
import json
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
import base64, io
import gc


yaml_file = "interpretation_config.yaml"
settings = yaml.safe_load(open(yaml_file))

interp_settings, model_settings, AEM_settings, det_inv_settings, stochastic_inv_settings, section_kwargs, crs = settings.values()

root = interp_settings['data_directory']

lines = interp_settings['lines']

# Prepare AEM data

em = aem_utils.AEM_data(name = AEM_settings['name'],
                        system_name = AEM_settings['system_name'],
                        netcdf_dataset = netCDF4.Dataset(os.path.join(root, AEM_settings['nc_path'])))

if AEM_settings["grid_sections"]:
    ## TODO add path checking function
    outdir = os.path.join(root, AEM_settings['gridding_params']['section_dir'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    em.interpolate_variables(variables=AEM_settings['gridding_params']['grid_vars'], lines=lines,
                             xres=AEM_settings['gridding_params']['xres'],
                             return_interpolated=False, save_to_disk=True,
                             output_dir = outdir)


# Prepare deterministic inversion

det = aem_utils.AEM_inversion(name = det_inv_settings['inversion_name'],
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, det_inv_settings['nc_path'])))

if det_inv_settings["grid_sections"]:
    ## TODO add path checking function
    outdir = os.path.join(root, det_inv_settings['gridding_params']['section_dir'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    det.grid_sections(variables = det_inv_settings['gridding_params']['grid_vars'], lines = lines,
                      xres = det_inv_settings['gridding_params']['xres'],
                      yres = det_inv_settings['gridding_params']['yres'],
                      return_interpolated = False, save_to_disk = True,
                      output_dir = outdir)
else:
    pass

# loaad layer grids
grid_file = os.path.join(root, det_inv_settings['layer_grid_path'])
det.load_lci_layer_grids_from_pickle(grid_file)

# Create polylines
det.create_flightline_polylines(crs = crs['projected'])

gdf_lines = det.flightlines[np.isin(det.flightlines['lineNumber'], lines)]

# Prepare stochastic inversion

rj = aem_utils.AEM_inversion(name = stochastic_inv_settings['inversion_name'],
                              inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, stochastic_inv_settings['nc_path'])))

if stochastic_inv_settings["grid_sections"]:
    ## TODO add path checking function
    outdir = os.path.join(root, stochastic_inv_settings['gridding_params']['section_dir'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    rj.grid_sections(variables = stochastic_inv_settings['gridding_params']['grid_vars'], lines = lines,
                      xres = stochastic_inv_settings['gridding_params']['xres'],
                      yres = stochastic_inv_settings['gridding_params']['yres'],
                      return_interpolated = False, save_to_disk = True,
                      output_dir = outdir)
else:
    pass


# To reduce the amount of data that is stored in memory the section data are stored as xarrays in pickle files
#  We will only bring them into memory as needed. Here we point the inversions to their pickle files

em.section_path = {}
det.section_path = {}
rj.section_path = {}
rj.distance_along_line = {}

for lin in lines:
    # Add path as attribute
    em.section_path[lin] = os.path.join(root, AEM_settings['gridding_params']['section_dir'],
                                        "{}.pkl".format(str(lin)))
    det.section_path[lin] = os.path.join(root, det_inv_settings['gridding_params']['section_dir'],
                                         "{}.pkl".format(str(lin)))
    rj.section_path[lin] = os.path.join(root, stochastic_inv_settings['gridding_params']['section_dir'],
                                        "{}.pkl".format(str(lin)))
    # Using this gridding we find the distance along the line for each garjmcmc site
    # Get a line mask
    line_mask = netcdf_utils.get_lookup_mask(lin, rj.data)
    # get the coordinates
    line_coords = rj.coords[line_mask]

    det_section_data = pickle2xarray(det.section_path[lin])

    dists = spatial_functions.xy_2_var(det_section_data,
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

        rj_section_data['grid_distances'] = spatial_functions.scale_distance_along_line(det_section_data, rj_section_data)
        # Save xarray back to pickle file

        xarray2pickle(rj_section_data, rj.section_path[lin])

    # Remove from memory
    rj_section_data = None
    det_section_data = None
    gc.collect()

## Annoying hack!!
rj.n_histogram_samples = np.sum(rj.data['log10conductivity_histogram'][0,0,:])


modelName = "Injune"

def initialise_strat_model(templateFile = "../data/surfaceTemplate.csv",
                           name = "Stratigraphic_model",
                           interp_file = None):
    model = modelling_utils.Statigraphic_Model(name = name, existing_interpretation_file = interp_file)

    model.template = pd.read_csv(templateFile, keep_default_na=False)

    model.initiatialise_surfaces_from_template(model.template)

    # Here we want to create surface and line options lists which we will need for our dropdown options
    surface_options = []

    for surfaceName in model.surfaces:
        surface_options.append({'label': surfaceName, 'value': surfaceName})

    model.surface_options = surface_options

    line_options = []

    for l in lines:
        line_options.append({'label': str(l), 'value': l})

    model.line_options = line_options

    return model

def subset_df_by_line(df_, line, line_col = 'SURVEY_LINE'):
    mask = df_[line_col] == line
    return df_[mask]

def style_from_surface(surfaceNames, styleName):
    style = []
    for item in surfaceNames:
        surface = getattr(model, item)
        prop = getattr(surface, styleName)
        style.append(prop)
    return style

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')), keep_default_na=False)
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))

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

    utm_coords = np.column_stack((xarr['easting'].values,xarr['northing'].values))

    df_ = subset_df_by_line(interpreted_points, line, line_col = line_col)

    dist, inds = spatial_functions.nearest_neighbours(df_[[easting_col,northing_col]].values,
                                                      utm_coords, max_distance=100.)

    grid_dists = xarr['grid_distances'].values[inds]
    elevs = df_[elevation_col].values

    surfName = df_['BoundaryNm'].values

    return grid_dists, elevs, surfName

def dash_conductivity_section(section, line, vmin, vmax, cmap, xarr):

    # Create subplots
    fig = make_subplots(rows=2, cols = 1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.2, 0.8])

    vmin = np.log10(vmin)
    vmax = np.log10(vmax)
    logplot = True

    # Extract data based on the section plot keyword argument
    if section == 'lci':
        #xarr = pickle2xarray(det.section_path[line])
        misfit = xarr['data_residual'].values
        z = np.log10(xarr['conductivity'].values)



    elif section.startswith('rj'):
        #xarr = pickle2xarray(rj.section_path[line])
        misfit = xarr['misfit_lowest'].values

        if section == "rj-p50":
            z = np.log10(xarr['conductivity_p50'])
        elif section == "rj-p10":
            z = np.log10(xarr['conductivity_p10'])
        elif section == "rj-p90":
            z = np.log10(xarr['conductivity_p90'])
        elif section == "rj-lpp":
            z = xarr['interface_depth_histogram'] / rj.n_histogram_samples
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
    # plot the data residual
    fig.add_trace(go.Scatter(x = dist_along_line,
                             y = misfit,
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = None),
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
                            hoverinfo=None,
                            ),
                      row = 2, col = 1,
        )


    # Add the elevation
    fig.add_trace(go.Scatter(x = xarr['grid_distances'].values,
                             y = xarr['elevation'].values,
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = 'skip'),
                  row = 2, col = 1,)

    df_rj_sites = rj.distance_along_line[line]

    labels = ["fiducial = " + str(x) for x in df_rj_sites['fiducial']]

    fig.add_trace(go.Scatter(x=df_rj_sites['distance_along_line'].values,
                             y= 20. + np.max(xarr['elevation'].values) * np.ones(shape=len(df_rj_sites),
                                                                                       dtype=np.float),
                             mode='markers',
                             hovertext=labels,
                             showlegend=False),
                  row=2, col=1)
    # Reverse y-axis
    fig.update_yaxes(autorange=True, row = 1, col = 1, title_text = "data residual")
    fig.update_yaxes(autorange=True, row=2, col=1, title_text="elevation (mAHD)")

    fig.update_xaxes(title_text= "distance along line " + " (m)", row=2, col=1)
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

    fig.add_trace(go.Scatter(x=grid_distances,
                             y=xarr['tx_height_measured'].values,
                             line=dict(color='black', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=1, col=1, )
    fig.add_trace(go.Scatter(x=grid_distances,
                             y=xarr['powerline_noise'].values,
                             line=dict(color='black', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=2, col=1, )
    # Add low moment data
    colours = ['green', 'aquamarine', 'darkslategrey', 'lightseagreen',
       'darkgrey', 'brown', 'rosybrown', 'white', 'greenyellow', 'khaki',
       'mediumturquoise', 'paleturquoise', 'bisque', 'saddlebrown',
       'chocolate', 'orchid', 'seashell', 'blue', 'burlywood',
       'deepskyblue', 'yellowgreen', 'lightgrey', 'turquoise']

    lm_data = xarr['low_moment_Z-component_EM_data'].values

    for j in range(lm_data.shape[1]):
        label = "low-moment gate " + str(j+1)
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=lm_data[:,j],
                                 mode = 'markers',
                                 marker={
                                         "color": colours[j],
                                         "size": 1.5
                                         },
                                 showlegend=False, hoverinfo='text',
                                 hovertext = label),
                      row=3, col=1, )
    hm_data = xarr['high_moment_Z-component_EM_data'].values
    for j in range(hm_data.shape[1]):
        label = "high-moment gate " + str(j + 1)
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=hm_data[:, j],
                                 mode = 'markers',
                                 marker={
                                         "color": colours[j],
                                         "size": 1.5
                                         },
                                 showlegend=False, hoverinfo='text',
                                 hovertext = label),
                      row=4, col=1, )
    fig.update_yaxes(title_text="tx height (m)", row=1, col=1)
    fig.update_yaxes(title_text="PLNI", type="log", row=2, col=1)
    fig.update_yaxes(title_text="LMZ data (V/Am^4)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="HMZ data (V/Am^4)", type="log", row=4, col=1)
    fig.update_xaxes(title_text="distance along line " + " (m)", row=4, col=1)
    fig['layout'].update({'height': 600})

    return fig

def plot_section_points(fig, line, section, df_interp, select_mask):
    if section == "lci":
        xarr = pickle2xarray(det.section_path[line])
    else:
        xarr = pickle2xarray(rj.section_path[line])
    # Create a scatter plot on the section using projection
    interpx, interpz, surfNames = interp2scatter(line, xarr, df_interp)

    if len(interpx) > 0:
        labels = ["surface = " + str(x) for x in surfNames]

        colours = style_from_surface(surfNames, "Colour")
        markers = style_from_surface(surfNames, "Marker")
        markerSize = [np.float(x) for x in style_from_surface(surfNames, "MarkerSize")]
        # To make selected points obvious
        if len(select_mask) > 0:
            for idx in select_mask:
                markerSize[idx] += 5.

        fig.add_trace(go.Scatter(x = interpx,
                        y = interpz,
                        mode = 'markers',
                        hovertext = labels,
                        marker = {"symbol": markers,
                                  "color": colours,
                                  "size": markerSize
                                  },
                        showlegend = True,
                                 xaxis= 'x2',
                                 yaxis='y2')
                      )
        fig.update_layout()
    return fig

def dash_pmap_plot(point_index):
    # Extract the data from the netcdf data
    D = netcdf_utils.extract_rj_sounding(rj, det,
                                         point_index)
    pmap = D['conductivity_pdf']
    x1,x2,y1,y2 = D['conductivity_extent']
    n_depth_cells, n_cond_cells  = pmap.shape

    x = np.linspace(x1,x2, n_cond_cells)
    y = np.linspace(y2,y1, n_depth_cells)

    fig = px.imshow(img = pmap,
                    x = x, y = y,
                    zmin = 0,
                    zmax = np.max(pmap),
                    aspect = 5,
                    color_continuous_scale = 'plasma')
    #  PLot the median, and percentile plots
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p10']),
                             y = D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'black',
                                     "width": 1.},
                             name = "p10 conductivity",
                             showlegend = False))
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p90']),
                             y = D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'black',
                                     "width": 1.},
                             name = "p90 conductivity",
                             showlegend = False))
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p50']),
                             y = D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'gray',
                                     "width": 1.,
                                     'dash': 'dash'},
                             name = "p50 conductivity",
                             showlegend = False))

    det_expanded, depth_expanded = plots.profile2layer_plot(D['det_cond'], D['det_depth_top'])

    fig.add_trace(go.Scatter(x=np.log10(det_expanded),
                             y= depth_expanded,
                             mode='lines',
                             line={"color": 'pink',
                                   "width": 1.,
                                   'dash': 'dash'},
                             name=det.name,
                             showlegend=False))

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

    xmin, xmax = np.min(rj.data['easting'][:]) - 500., np.max(rj.data['easting'][:]) + 500.
    ymin, ymax = np.min(rj.data['northing'][:]) - 500., np.max(rj.data['northing'][:]) + 500.

    fig.update_layout(yaxis=dict(range=[ymin, ymax],
                                 scaleanchor = 'x',
                                 scaleratio = 1.),
                      xaxis=dict(range=[xmin, xmax]))
    fig['data'][0]['showscale'] = False
    return fig


stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

model = initialise_strat_model(name = modelName, interp_file = model_settings['interpFile'],
                               templateFile=model_settings['templateFile'])

app.layout = html.Div([
    html.Div(
                [
                    html.Div(html.H1("AEM interpretation dash board"),
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
                                            options=model.surface_options,
                                            value=(model.surface_options[0]['label']))

                             ],className = "three columns"),
                    html.Div([html.H4("Select section"),
                             dcc.Dropdown(id = "section_dropdown",
                                            options=[
                                                    {'label': 'laterally constrained inversion',
                                                     'value': 'lci'},
                                                    {'label': 'garjmcmctdem - p50',
                                                     'value': 'rj-p50'},
                                                    {'label': 'garjmcmctdem - p10',
                                                     'value': 'rj-p10'},
                                                    {'label': 'garjmcmctdem - p90',
                                                     'value': 'rj-p90'},
                                                    {'label': 'garjmcmctdem - layer probability',
                                                     'value': 'rj-lpp'}],
                                            value="lci"),

                             ],className = "three columns"),
                    html.Div([html.H4("Select line"),
                             dcc.Dropdown(id = "line_dropdown",
                                            options=model.line_options,
                                            value= int(model.line_options[0]['label'])),
                             ],className = "three columns")
                ], className = 'row'
            ),
    html.Div(
            [
                html.Div(html.Pre(id='click-data'),
                         className = "three columns"),
                html.Div(dash_table.DataTable(id='surface_table',
                                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                              columns = [{"name": i, "id": i} for i in model.template.columns],
                                              data=model.template.to_dict('records'),
                                                editable=True,
                                            fixed_columns={ 'headers': True},
                                            sort_action="native",
                                            sort_mode="multi",
                                            row_selectable=False,
                                            row_deletable=False,
                                              style_table={
                                                  'maxHeight': '400px',
                                                  'overflowY': 'scroll',
                                                  'maxWidth': '500px',
                                                  'overflowX': 'scroll'}
                                              ),
                         className = "three columns"),
                html.Div([html.Div(["Conductivity plotting minimum: ", dcc.Input(
                                    id="vmin", type="number",
                                    min=0.001, max=10, value = section_kwargs['vmin'])],
                         className = 'row'),
                         html.Div(["Conductivity plotting maximum: ", dcc.Input(
                                    id="vmax", type="number",
                                    min=0.001, max=10, value = section_kwargs['vmax'])],
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
        )], style={'height': '600px'}),
        ]),
    html.Div([html.Div(
        html.Div([
            dash_table.DataTable(id='interp_table',
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                columns = [{"name": i, "id": i} for i in model.interpreted_points[model.interpreted_points['SURVEY_LINE'] == int(model.line_options[1]['label'])]],
                                data=model.interpreted_points[model.interpreted_points['SURVEY_LINE'] == int(model.line_options[0]['label'])].to_dict('records'),
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
    dcc.Store(id='interp_memory')

])

# megacallback function for updating the interpreted points. This is either done by clicking on the section or deleting
# from the table
@app.callback(
    [Output('interp_memory', 'data'),
     Output('section_plot', 'figure'),
     Output('interp_table', 'data')],
    [Input('section_plot', 'clickData'),
     Input('interp_table', 'data_previous'),
     Input('section_dropdown', 'value'),
     Input('section_tabs', 'value'),
     Input("line_dropdown", 'value'),
     Input('vmin', 'value'),
     Input('vmax', 'value')],
     [State('interp_table', 'data'),
      State("surface_dropdown", 'value'),
      State("interp_memory", 'data')])
def update_interp_table(clickData, previous_table, section, section_tab, line, vmin, vmax, current_table,
                        surfaceName, interpreted_points):
    trig_id = find_trigger()
    # Access the data from the store. The or is in case of None callback at initialisation
    interpreted_points = interpreted_points or model.interpreted_points.to_dict('records')
    df = pd.DataFrame(interpreted_points)
    if section == "lci":
        xarr = pickle2xarray(det.section_path[line])
    else:
        xarr = pickle2xarray(rj.section_path[line])


    if np.logical_and(trig_id == 'section_plot.clickData', section_tab == 'conductivity_section'):
        if clickData['points'][0]['curveNumber'] == 1:
            # Get the interpretation data from the click function
            surface = getattr(model, surfaceName)


            eventxdata, eventydata = clickData['points'][0]['x'], clickData['points'][0]['y']
            min_idx = np.argmin(np.abs(xarr['grid_distances'].values - eventxdata))

            easting = xarr['easting'].values[min_idx]
            northing = xarr['northing'].values[min_idx]
            elevation = xarr['elevation'].values[min_idx]
            depth = elevation - eventydata
            fid = xy2fid(easting, northing, det)

            # append to the surface object interpreted points
            interp = {'fiducial': fid,
                      'inversion_name': section,
                      'X': np.round(easting, 0),
                      'Y': np.round(northing, 0),
                      'DEPTH': np.round(depth, 0),
                      'ELEVATION': np.round(eventydata,1),
                      'DEM': np.round(elevation,1),
                      'UNCERTAINTY': np.nan,  # TODO implement FULL WIDTH HALF MAXIMUM
                      'Type': surface.Type,
                      'BoundaryNm': surface.name,
                      'BoundConf': surface.BoundConf,
                      'BasisOfInt': surface.BasisOfInt,
                      'OvrConf': surface.OvrConf,
                      'OvrStrtUnt': surface.OvrStrtUnt,
                      'OvrStrtCod': surface.OvrStrtCod,
                      'UndStrtUnt': surface.UndStrtUnt,
                      'UndStrtCod': surface.UndStrtCod,
                      'WithinType': surface.WithinType,
                      'WithinStrt': surface.WithinStrt,
                      'WithinStNo': surface.WithinStNo,
                      'WithinConf': surface.WithinConf,
                      'InterpRef': surface.InterpRef,
                      'Comment': surface.Comment,
                      'SURVEY_LINE': line,
                      'Operator': surface.Operator,
                      "point_index": min_idx
                      }
            df_new = pd.DataFrame(interp, index=[0])
            df = pd.DataFrame(interpreted_points).append(df_new)
        else:
            print('nil')
            raise PreventUpdate
            return

    elif trig_id == 'interp_table.data_previous':
        if previous_table is None:
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
    if section_tab == 'conductivity_section':
        fig = dash_conductivity_section(section, line,
                                        vmin=vmin,
                                        vmax=vmax,
                                        cmap=section_kwargs['cmap'],
                                        xarr = xarr)
    elif section_tab == 'data_section':
        fig = dash_EM_section(line)
    # Subset to the
    df_ss = df[df['SURVEY_LINE'] == line]
    if np.logical_and(len(df_ss) > 0, section_tab == 'conductivity_section'):
        fig = plot_section_points(fig, line, section, df_ss, select_mask=[])

    return df.to_dict('records'), fig, df_ss.to_dict('records')


# Render map plot
@app.callback(Output('map-tabs-content', 'children'),
              [Input('map_tabs', 'value'),
               Input("line_dropdown", 'value'),
               Input('vmin', 'value'),
               Input('vmax', 'value'),
               Input('layerGrid', 'value'),
               Input('section_plot', 'clickData')])
def update_tab(tab, line, vmin, vmax, layer, clickData):
    if tab == 'map_plot':
        fig = flightline_map(line, vmin, vmax, layer)
        return html.Div([
            dcc.Graph(
                id='polylines',
                figure=fig
            ),
        ])
    elif tab == 'pmap_plot':
        if clickData is not None:
            if clickData['points'][0]['curveNumber'] == 3:
                point_idx = clickData['points'][0]['pointIndex']
                fig = dash_pmap_plot(point_idx)
                return html.Div([
                    dcc.Graph(
                        id='pmap_plot',
                        figure=fig
                    ),
                ])
        else:
            return html.Div(["Click on a purple point from the conductivity section to view the probability maps."])

@app.callback(
    Output('export_message', 'children'),
    [Input("export", 'n_clicks')],
    [State('export-path', 'value'),
    State('interp_memory', 'data')])
def export_data_table(nclicks, value, interpreted_points):
    interpreted_points = interpreted_points or model.interpreted_points
    if np.logical_and(nclicks > 0, value is not None):
        if os.path.exists(os.path.dirname(value)):
            interpreted_points.reset_index(drop=True).to_csv(value)
            return "Successfully exported to " + value
        else:
            return value + " is an invalid file path."

app.run_server(debug = True)

'''

@app.callback([Output("surface_dropdown", 'value'),
               Output("surface_dropdown", 'options')],
              [Input('template-upload', 'contents')],
              [State('template-upload', 'filename'),
               State("interp_memory", "data")])
def update_output(contents, filename, interpreted_points):
    if contents is None:
        return model.surfaces[0], model.surface_options
    df = parse_contents(contents, filename)

    valid = True#check_validity(df)
    ##TODO write a file validity function

    if valid:
        # If interpreted points is empty
        if len(pd.DataFrame(interpreted_points)) < 1:
            ## TODO create function for this
            for item in model.surfaces:
                delattr(model, item)
            model.surfaces = []
            model.surface_options = []
            model.template = None
            for index, row in df.iterrows():
                model.initiatialise_surface(pd.Series(row))
            for surfaceName in model.surfaces:
                model.surface_options.append({'label': surfaceName, 'value': surfaceName})
        else:
            for index, row in df.iterrows():
                surfaceName = row['SurfaceName']
                idx = model.template[model.template['SurfaceName'] == surfaceName].index
                df_row = pd.DataFrame(row, index=idx)
                if not df_row.equals(model.template.iloc[idx]):
                    model.template.at[idx, :] = df_row
                    # update our model surface attributes
                    delattr(model, surfaceName)
                    model.initiatialise_surface(pd.Series(row))
        model.template = df.copy()
        ## TODO remove

        return model.template.values[0], model.surface_options
    else:
        raise Exception("Input file is not valid")

@app.callback(
    Output('export_message', 'children'),
    [Input("export", 'n_clicks')],
    [State('export-path', 'value'),
    State('interp_memory', 'data')])
def export_data_table(nclicks, value, interpreted_points):

    if np.logical_and(nclicks > 0, value is not None):
        if os.path.exists(os.path.dirname(value)):
            interpreted_points.reset_index(drop=True).to_csv(value)
            return "Successfully exported to " + value
        else:
            return value + " is an invalid file path."



@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value'),
               Input("line_dropdown", 'value'),
               Input('vmin', 'value'),
               Input('vmax', 'value'),
               Input('layerGrid', 'value'),
               Input('section_plot', 'clickData')])
def update_tab(tab, line, vmin, vmax, layer, clickData):
    if tab == 'map_plot':
        fig = flightline_map(line, vmin, vmax, layer)
        return html.Div([
            dcc.Graph(
                id='polylines',
                figure=fig
            ),
        ])
    elif tab == 'pmap_plot':
        if clickData is not None:
            if clickData['points'][0]['curveNumber'] == 3:
                point_idx = clickData['points'][0]['pointIndex']
                fig = dash_pmap_plot(point_idx)
                return html.Div([
                    dcc.Graph(
                        id='pmap_plot',
                        figure=fig
                    ),
                ])



@app.callback(Output('output', 'children'),
              [Input("surface_dropdown", 'value'),
               Input('interp_table', 'data_previous')],
              [State('interp_table', 'data')])
def show_removed_rows(surfaceName, previous, current):
    if previous is None:
        dash.exceptions.PreventUpdate()
    else:
        fids = []
        for row in previous:
            if row not in current:
                fids.append(row['fiducial'])
                # Remove from dataframe
        model.interpreted_points = model.interpreted_points[~model.interpreted_points['fiducial'].isin(fids)]
        return [f'Just removed fiducial : {fids}']

@app.callback(
    Output('surface_table', 'data'),
    [Input("surface_dropdown", 'value'),
     Input('surface_table', 'data_timestamp')],
    [State('surface_table', 'data')])
def update_model(surfaces, timestamp, rows):
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    # Check if all the surface are the same
    df_rows = pd.DataFrame(rows)
    # First case is where we have changed the surfaces entirely
    if trig_id == "surface_dropdown.value":
        rows = model.template.to_dict("records")
    elif trig_id == "surface_table.data_timestamp":
        for row in rows:
            surfaceName= row['SurfaceName']
            idx = model.template[model.template['SurfaceName'] == surfaceName].index
            df_row = pd.DataFrame(row, index = idx)
            if not df_row.equals(model.template.iloc[idx]):
                model.template.at[idx, :] = df_row
                # update our model surface attributes
                delattr(model, surfaceName)
                model.initiatialise_surface(pd.Series(row))
    return rows
'''

