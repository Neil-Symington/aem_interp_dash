# Import python modules
import numpy as np
import yaml
import netCDF4
import os
from dash.exceptions import PreventUpdate
from garjmcmctdem_utils import spatial_functions, aem_utils
from garjmcmctdem_utils.misc_utils import pickle2xarray, xarray2pickle
import warnings
import pandas as pd

from shapely import wkt
warnings.filterwarnings('ignore')
# Dash dependencies

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


#profiling
import time
from functools import wraps
def profile(f):
    @wraps(f)
    def profiling_wrapper(*args, **kwargs):
        tic = time.perf_counter()
        try:
            res = f(*args, **kwargs)
            toc = time.perf_counter()
            print("Call to", f.__name__, "took", toc - tic, "seconds")
        except PreventUpdate:
            toc = time.perf_counter()
            print("Call to", f.__name__, "took", toc - tic, "seconds and did not update")
            raise PreventUpdate
        return res
    return profiling_wrapper

yaml_file = "interpretation_config.yaml"
settings = yaml.safe_load(open(yaml_file))

interp_settings, model_settings, inversion_settings, section_settings,\
borehole_settings, crs = settings.values()

root = interp_settings['data_directory']

lines = interp_settings['lines']

# Prepare AEM data
det = aem_utils.AEM_inversion(name = inversion_settings['inversion_name'],
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, inversion_settings['nc_path'])))

# Grid the data if the user wants
if inversion_settings["grid_sections"]:
    print("Gridding AEM data. This may take a few minutes.")

    outdir = os.path.join(root, inversion_settings['section_directory'])
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    det.grid_sections(geometry_variables = inversion_settings['geometry_variables'],
                      inversion_variables = inversion_settings['inversion_variables'],
                      data_variables = inversion_settings['data_variables'],
                      lines = lines,  xres = inversion_settings['horizontal_resolution'],
                      yres = inversion_settings['vertical_resolution'],
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

det.section_path = {}

# Iterate through the lines
for lin in lines:

    det.section_path[lin] = os.path.join(root, inversion_settings['section_directory'],
                                         "{}.pkl".format(str(lin)))
    ## TODO write a function for read borehole data

# Define colour stretch for em data

viridis = cm.get_cmap('plasma')
n_gates = det.data.dimensions['window'].size

lm_colours = [matplotlib.colors.rgb2hex(x) for x in viridis(np.linspace(0, 1, n_gates))]


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

# section functions
def xy2data(x,y, dataset):
    dist, ind = spatial_functions.nearest_neighbours([x, y],
                                                     dataset.coords,
                                                     max_distance = 100.)
    return dataset.data['x_secondary_field_observed'][ind][0], dataset.data['z_secondary_field_observed'][ind][0],\
           dataset.data['x_secondary_field_predicted'][ind][0], dataset.data['z_secondary_field_predicted'][ind][0]

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

def dash_conductivity_section(vmin, vmax, cmap, xarr):

    # Create subplots
    fig = make_subplots(rows=6, cols = 1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.1, 0.1, 0.1, 0.34, 0.35, 0.01])
    vmin = np.log10(vmin)
    vmax = np.log10(vmax)

    # Extract data based on the section plot keyword argument
    misfit = xarr['phid'].values
    z = np.log10(xarr['conductivity'].values)

    tickvals = np.linspace(vmin, vmax, 5)
    ticktext = [str(np.round(x, 3)) for x in 10**tickvals]

    dist_along_line = xarr['grid_distances'].values
    grid_elevations = xarr['grid_elevations'].values
    elevation = xarr['elevation'].values
    easting = xarr['easting'].values
    northing = xarr['northing'].values

    # Subset based on line length to avoid long rendering for plots
    #ss = np.int(math.ceil(np.max(dist_along_line) / 10000.))  ##TODO move subsetting to the yaml file
    grid_distances = dist_along_line[:]
    tx_height = xarr['tx_height'].values[:]
    data_observed = np.arcsinh(np.sqrt(xarr['x_secondary_field_observed'].values**2 + xarr['z_secondary_field_observed'].values**2))
    data_predicted = np.arcsinh(np.sqrt(xarr['x_secondary_field_predicted'].values**2 + xarr['z_secondary_field_predicted'].values**2))
    txrx_dx = xarr['txrx_dx'].values
    txrx_dz = xarr['txrx_dz'].values
    inv_txrx_dx = xarr['inverted_txrx_dx'].values
    inv_txrx_dz = xarr['inverted_txrx_dz'].values


    fig.add_trace(go.Scatter(x=grid_distances,
                             y=txrx_dx,
                             line=dict(color='red', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=1, col=1, )

    fig.add_trace(go.Scatter(x=grid_distances,
                             y=txrx_dz,
                             line=dict(color='blue', width=1),
                             showlegend=False, hoverinfo='skip'),
                  row=2, col=1, )
    fig.add_trace(go.Scatter(x=grid_distances,
                             y=inv_txrx_dx,
                             line=dict(color='black', width=0.8),
                             showlegend=False, hoverinfo='skip'),
                  row=1, col=1, )

    fig.add_trace(go.Scatter(x=grid_distances,
                             y=inv_txrx_dz,
                             line=dict(color='pink', width=0.8),
                             showlegend=False, hoverinfo='skip'),
                  row=2, col=1, )


    # plot the data residual
    fig.add_trace(go.Scatter(x=dist_along_line,
                             y=misfit,
                             line=dict(color='black', width=3),
                             showlegend=False, hoverinfo=None,
                             name="residual"),
                  row=3, col=1)

    # prepare the line data

    for j in range(data_observed.shape[1]):
        labels = ["Vector sum gate " + str(j)]
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=data_observed[:, j],
                                 mode='lines',
                                 line={
                                     "color": 'black'
                                 },
                                 showlegend=False, hoverinfo="none"),
                      row=4, col=1, )
        fig.add_trace(go.Scatter(x=grid_distances,
                                 y=data_predicted[:, j],
                                 mode='lines',
                                 line={
                                     "color": lm_colours[j]
                                 },
                                 hovertext=labels,
                                 showlegend=False, hoverinfo="none"),
                      row=4, col=1, )

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
                      row = 5, col = 1
        )

    # Add the elevation
    fig.add_trace(go.Scatter(x = dist_along_line,
                             y = elevation,
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = 'skip'),
                  row = 5, col = 1,)

    fig.add_trace(go.Scatter(x=dist_along_line,
                             y=tx_height + elevation,
                             line=dict(color='blue', width=1),
                             showlegend=False, hoverinfo='text',
                             text = "tx height"),
                  row=5, col=1, )

    # Create a list of easting and northing (strings) for label
    labels = []
    for i in range(len(easting)):
        labels.append(' '.join(["x:", "{:.1f}".format(easting[i]), "y:", "{:.1f}".format(northing[i])]))
    # Add the easting/ northing as a scatter plot
    fig.add_trace(go.Scatter(x=dist_along_line,
                             y=np.ones(len(dist_along_line)),
                             line=dict(color='black', width=3),
                             hovertext = labels,
                             showlegend=False,
                             name = "coordinates"),
                  row=6, col=1, )

    # Reverse y-axis
    fig.update_yaxes(autorange=True, row = 1, col = 1, title_text = "elevation (mAHD)")
    fig.update_yaxes(autorange=True, row=2, col=1, title_text="elevation (mAHD)")
    fig.update_yaxes(autorange=True, row=3, col=1, title_text="PhiD", type='log')
    fig.update_yaxes(autorange=True, row=4, col=1, title_text="asinh(fT)")
    fig.update_yaxes(autorange=True, row=5, col=1, title_text="elevation (mAHD)")
    fig.update_yaxes(visible=False, showticklabels=False, row=6, col=1)

    fig.update_xaxes(title_text= "distance along line " + " (m)", row=3, col=1)
    fig['layout'].update({'height': 1000})
    return fig

def aem_data_plot(xobs, zobs, xpred,zpred):
    multiplicative_noise= 0.002

    Hx_add_noise = np.array([0.010619, 0.009453, 0.008506, 0.006687, 0.007244, 0.005554, 0.004701, 0.004353, 0.003539,
                             0.003493, 0.003035, 0.002875, 0.002343, 0.001613, 0.001304])

    Hz_add_noise = np.array([0.005554, 0.005280, 0.004101, 0.003093, 0.002969, 0.002723, 0.002696, 0.002429, 0.002377,
                             0.002188, 0.002018, 0.001818, 0.001557, 0.001106, 0.000906])

    window_times = np.array([1.15470343e-05, 3.94405263e-05, 6.63324807e-05, 1.04774925e-04, 1.70098025e-04,
                             2.73495887e-04, 4.42166350e-04, 7.03483404e-04, 1.09542484e-03, 1.69410746e-03,
                             2.63014149e-03, 4.09949049e-03, 6.40149636e-03, 9.96138991e-03, 1.57496229e-02])
    fig = go.Figure()

    dat_err_x = Hx_add_noise + xobs * multiplicative_noise
    dat_err_z = Hz_add_noise + zobs * multiplicative_noise

    fig.add_trace(go.Scatter(x=window_times,
                             y=xobs,
                             #error_y = dat_err_x,
                             mode='markers',
                             marker_size=3.,
                             showlegend=False))
    fig.add_trace(go.Scatter(x=window_times,
                             y=zobs,
                             #error_y=dat_err_z,
                             mode='markers',
                             marker_size=3.,
                             showlegend=False))
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
                                 xaxis= 'x5',
                                 yaxis='y5')
                      )
        fig.update_layout()
    return fig

def dataFitMap():

    fig = go.Figure()

    return fig


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
df_interpreted_points['MarkerSize'] = marker_size

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
                                                    {'label': 'layered earth inversion',
                                                     'value': 'galei'}],
                                            value="galei"),

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
                                    min=0.0001, max=10, value = section_settings['vmin'])],
                         className = 'row'),
                         html.Div(["Conductivity plotting maximum: ", dcc.Input(
                                    id="vmax", type="number",
                                    min=0.001, max=10, value = section_settings['vmax'])],
                         className='row')
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
                    dcc.Tab(label='User exercise', value='conductivity_section'),
                    dcc.Tab(label='GA interpretation', value='data_section'),
                 ]),
        html.Div([dcc.Graph(
            id='section_plot',
        )], style={'height': '1000'}),
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
                dcc.Tab(label='Data plot', value='data_plot'),
            ]),
            html.Div(id='map-tabs-content', style = {'height': '600px'})
        ],
            className = "six columns")],
        className = 'row'),
    html.Div(id = 'output', style={'display': 'none'}),
    dcc.Store(id='interp_memory'),
    dcc.Store(id = "model_memory"), # storage for model template file

])

# megacallback function for updating the interpreted points. This is either done by clicking on the section or deleting
# from the table
@app.callback(
    [Output('interp_memory', 'data'),
     Output('section_plot', 'figure'),
     Output('interp_table', 'data'),
     Output('map-tabs-content', 'figure')],
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
      State("model_memory", "data")])
def update_many(clickData, previous_table, section, section_tab, line, vmin, vmax, selected_rows, current_table,
                        surfaceName, interpreted_points, model):
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

    xarr = pickle2xarray(det.section_path[line])

    figgy = None #TODO fix this
  
    if trig_id == 'section_plot.clickData' and section_tab == 'conductivity_section':

        if clickData['points'][0]['curveNumber'] == 35:
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

            fid = xy2fid(easting, northing, det)
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

        elif clickData['points'][0]['curveNumber'] < 35:
            eventxdata, eventydata = clickData['points'][0]['x'], clickData['points'][0]['y']
            min_idx = np.argmin(np.abs(xarr['grid_distances'].values - eventxdata))
            easting = xarr['easting'].values[min_idx]
            northing = xarr['northing'].values[min_idx]
            # new get the data
            xobs, zobs, xpred, zpred = xy2data(easting, northing, det)
            figgy = aem_data_plot(xobs, zobs, xpred, zpred)
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
        fig = dash_conductivity_section(vmin=vmin,
                                        vmax=vmax,
                                        cmap=section_settings['cmap'],
                                        xarr = xarr)


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
            # plot the boreholes as segments
        #    fig = plot_borehole_segments(fig, df_bh_ss)

        fig['layout'].update({'uirevision': line})

    elif section_tab == 'data_section':
        ## TODO put in function and add precanned interpretation
        fig = dash_conductivity_section(vmin=vmin,
                                        vmax=vmax,
                                        cmap=section_settings['cmap'],
                                        xarr=xarr)

        # Now plot boreholes
        # df_bh_ss = df_bh[df_bh['line'] == line]
        # if len(df_bh_ss) > 0:
        # plot the boreholes as segments
        #    fig = plot_borehole_segments(fig, df_bh_ss)

        fig['layout'].update({'uirevision': line})

    return df.to_dict('records'), fig, df_ss.to_dict('records'), figgy


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
    return df_model['SurfaceName'][0], list2options(df_model['SurfaceName'].values),\
           model_store, [{"name": i, "id": i} for i in df_model.columns], model_store

app.run_server(debug = True)

