# Import python modules
import numpy as np
import geopandas as gpd
import netCDF4
import sys, os
sys.path.append("../scripts")

import spatial_functions
import aem_utils
import netcdf_utils
import modelling_utils
import plotting_functions as plots
import warnings
import pandas as pd
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


modelName = "Injune"

root ="/home/nsymington/Documents/GA/dash_data"
# path to determinist netcdf file
det_nc_path = os.path.join(root, "Injune_lci_MGA55.nc")
# path to dertiminist grid
det_grid_path = os.path.join(root, "Injune_layer_grids.p")

grid_lci = False #If the lci conductivity sections have not yet been gridded then make this flag true
# path to rjmcmcmtdem pmap file
rj_nc_path = os.path.join(root, "Injune_rjmcmc_pmaps.nc")

grid_rj = False #If the rj conductivity sections have not yet been gridded then make this flag true

project_crs = 'EPSG:28353'
lines = [200101, 200401, 200501, 200601, 200701, 200801,
         200901, 201001, 201101, 201201, 201301, 201401, 201501,
         201601, 201701, 201801, 201901, 202001, 202101, 202201,
         202301, 202401, 202501, 202601, 202701, 202801, 912011]

# Create an instance
lci = aem_utils.AEM_inversion(name = 'Laterally Contrained Inversion (LCI)',
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(det_nc_path))

# Run function
lci.load_lci_layer_grids_from_pickle(det_grid_path)

# Create instance
rj = aem_utils.AEM_inversion(name = 'GARJMCMCTDEM',
                             inversion_type = 'stochastic',
                             netcdf_dataset = netCDF4.Dataset(rj_nc_path))


# Now we have the lines we can grid the lci conductivity data onto vertical grids (known as sections)
# this is the easiest way to visualise the AEM conuctivity in 2-dimensions

# Assign the lci variables to grid
grid_vars = ['conductivity', 'data_residual', 'depth_of_investigation']


# Define the resolution of the sections
xres, yres = 40., 5.

# We will use the lines from the rj



# Define the output directory if saving the grids as hdf plots

hdf5_dir = os.path.join(root, "hdf5_lci")

# if the directory doesn't exist, then create it
if not os.path.exists(hdf5_dir):
    os.mkdir(hdf5_dir)

if grid_lci:
    lci.grid_sections(variables = grid_vars, lines = lines, xres = xres, yres = yres,
                      return_interpolated = True, save_hdf5 = True, hdf5_dir = hdf5_dir)
else:
    lci.load_sections_from_file(hdf5_dir, grid_vars, lines = lines)

# Grid the rj sections

# Assign the lci variables to grid
grid_vars = ['conductivity_p10', 'conductivity_p50', 'conductivity_p90', 'interface_depth_histogram',
             'misfit_lowest', 'misfit_average']

# Define the resolution of the sections
xres, yres = 50., 2.

# Define the output directory if saving the grids as hdf plots

hdf5_dir = os.path.join(root, "hdf5_rj")

# if the directory doesn't exist, then create it
if not os.path.exists(hdf5_dir):
    os.mkdir(hdf5_dir)

if grid_rj:
    rj.grid_sections(variables = grid_vars, lines = lines, xres = xres, yres = yres,
                     return_interpolated = True, save_hdf5 = True, hdf5_dir = hdf5_dir)
else:
    rj.load_sections_from_file(hdf5_dir, grid_vars, lines = lines)

# Create polylines
lci.create_flightline_polylines()

gdf_lines = gpd.GeoDataFrame(data = {'lineNumber': lci.flight_lines.keys(),
                                     'geometry': lci.flight_lines.values()},
                             geometry= 'geometry',
                             crs = project_crs)

gdf_lines = gdf_lines[np.isin(gdf_lines['lineNumber'], lines)]

## TODO wrap this up in a function
# Using this gridding we find the distance along the line for each site
# Iterate through the lines
rj.distance_along_line = {}

for lin in lines:
    # Get a line mask
    line_mask = netcdf_utils.get_lookup_mask(lin, rj.data)
    # get the coordinates
    line_coords = rj.coords[line_mask]

    dists = spatial_functions.xy_2_var(lci.section_data[lin],
                                      line_coords,
                                      'grid_distances')
    easting = spatial_functions.xy_2_var(lci.section_data[lin],
                                      line_coords,
                                      'easting')
    northing = spatial_functions.xy_2_var(lci.section_data[lin],
                                      line_coords,
                                      'northing')
    # Add a dictionary with the point index distance along the line to our inversion instance
    rj.distance_along_line[lin] = pd.DataFrame(data = {"point_index": np.where(line_mask)[0],
                                                       "distance_along_line": dists,
                                                       'easting': easting,
                                                       'northing': northing,
                                                       'fiducial': rj.data['fiducial'][line_mask]}
                                               ).set_index('point_index')


# Setup the model

headings = ["fiducial", "inversion_name",'X', 'Y', 'ELEVATION', "DEM", "DEPTH", "UNCERTAINTY", "Type",
            "BoundaryNm", "BoundConf", "BasisOfInt", "OvrConf", "OvrStrtUnt", "OvrStrtCod", "UndStrtUnt",
           "UndStrtCod", "WithinType", "WithinStrt", "WithinStNo", "WithinConf", "InterpRef",
            "Comment", "SURVEY_LINE", "Operator"]


def initialise_strat_model(templateFile = "../data/blankSurfaceTemplate.csv",
                           name = "Stratigraphic_model"):
    model = modelling_utils.Statigraphic_Model(name = name, interpreted_point_headings = headings)

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

def interp2scatter(line, gridded_data, interpreted_points, easting_col = 'X',
                   northing_col = 'Y', elevation_col = 'ELEVATION',
                   line_col = 'SURVEY_LINE'):

    utm_coords = np.column_stack((gridded_data[line]['easting'],
                                  gridded_data[line]['northing']))

    df_ = subset_df_by_line(interpreted_points, line, line_col = line_col)

    dist, inds = spatial_functions.nearest_neighbours(df_[[easting_col,northing_col]].values,
                                                      utm_coords, max_distance=100.)

    grid_dists = gridded_data[line]['grid_distances'][inds]
    elevs = df_[elevation_col].values

    surfName = df_['BoundaryNm'].values

    return  grid_dists, elevs, surfName

def dash_section(line, df_interp, select_mask, vmin, vmax, cmap):
    # Create subplots
    fig = make_subplots(rows=2, cols = 1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.2, 0.8])

    # plot the data residual
    if section_kwargs['section_plot'] == "lci":
        section_data = lci.section_data
        fig.add_trace(go.Scatter(x = section_data[line]['grid_distances'],
                                 y = section_data[line]['data_residual'],
                                 line=dict(color='black', width=3),
                                 showlegend = False, hoverinfo = None),
                       row = 1, col = 1,)
    else:
        section_data = rj.section_data
        fig.add_trace(go.Scatter(x = section_data[line]['grid_distances'],
                                 y = np.log10(section_data[line]['misfit_lowest']),
                                 line=dict(color='black', width=3),
                                 showlegend = False, hoverinfo = None),
                       row = 1, col = 1,)

    # Create the grid
    if section_kwargs['section_plot'] == "lci":
        fig.add_trace(go.Heatmap(z = np.log10(section_data[line]['conductivity']),
                        zmin = np.log10(vmin),
                        zmax = np.log10(vmax),
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale =cmap,
                        hoverinfo=None,
                        ),
                      row = 2, col = 1,
        )
    elif section_kwargs['section_plot'] == "rj-p50":
        fig.add_trace(go.Heatmap(z = np.log10(section_data[line]['conductivity_p50']),
                        zmin = np.log10(vmin),
                        zmax = np.log10(vmax),
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale =cmap,
                        hoverinfo=None,
                        ),
                      row = 2, col = 1,
        )
    elif section_kwargs['section_plot'] == "rj-p10":
        fig.add_trace(go.Heatmap(z = np.log10(section_data[line]['conductivity_p10']),
                        zmin = np.log10(vmin),
                        zmax = np.log10(vmax),
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale =cmap,
                        hoverinfo=None,
                        ),
                      row = 2, col = 1,
        )
    elif section_kwargs['section_plot'] == "rj-p90":
        fig.add_trace(go.Heatmap(z = np.log10(section_data[line]['conductivity_p90']),
                        zmin = np.log10(vmin),
                        zmax = np.log10(vmax),
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale =cmap,
                        hoverinfo=None,
                        ),
                      row = 2, col = 1,
        )
    elif section_kwargs['section_plot'] == 'rj-conf':
        ##TODO decide on different metric for confidence

        confidence = plots.percentiles2pnci(section_data[line]['conductivity_p10'],
                                            section_data[line]['conductivity_p90'],
                                            upper_threshold = 0.99,
                                            lower_threshold = 0.01)

        fig.add_trace(go.Heatmap(z = confidence,
                        zmin = 0.1,
                        zmax = 0.9,
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale ="YlGn"
                        ),
                      row = 2, col = 1,
        )
    elif section_kwargs['section_plot'] == "rj-lpp":

        fig.add_trace(go.Heatmap(z = section_data[line]['interface_depth_histogram']/rj.data.nsamples,
                        zmin = 0.01,
                        zmax = 0.99,
                        x = section_data[line]['grid_distances'],
                        y = section_data[line]['grid_elevations'],
                        colorscale ="greys",
                        ),
                      row = 2, col = 1,
        )

    # Add the elevation
    fig.add_trace(go.Scatter(x = section_data[line]['grid_distances'],
                             y = section_data[line]['elevation'],
                             line=dict(color='black', width=3),
                             showlegend = False, hoverinfo = 'skip'),
                  row = 2, col = 1,)

    # Now we add the rjmcmc sites to the section

    df_rj_sites = rj.distance_along_line[line]

    labels = ["fiducial = " + str(x) for x in df_rj_sites['fiducial']]

    fig.add_trace(go.Scatter(x = df_rj_sites['distance_along_line'].values,
                    y = 20. +np.max(section_data[line]['elevation'])*np.ones(shape = len(df_rj_sites),
                                                                        dtype = np.float),
                            mode = 'markers',
                            hovertext = labels,
                            showlegend = False),
                row = 2, col = 1)

    if len(df_interp) > 1:
        if np.logical_or(section_kwargs['section_plot'] == "rj-p50",
                         section_kwargs['section_plot'] == "lci"):
            # Get the ticks
            tickvals = np.linspace(np.log10(section_kwargs['vmin']),
                                    np.log10(section_kwargs['vmax']),
                                    5)

            ticktext = [str(np.round(x,3)) for x in 10**tickvals]

            fig.update_layout(coloraxis_colorbar=dict(
            title="conductivity",
            tickvals=tickvals,
            ticktext=ticktext,
            ))


        interpx, interpz, surfNames = interp2scatter(line, section_data, df_interp)

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
                            showlegend = True),
                            row = 2, col = 1
                          )

    # Reverse y-axis
    fig.update_yaxes(autorange=True)

    return fig

def dash_pmap_plot(point_index):
    # Extract the data from the netcdf data
    D = netcdf_utils.extract_rj_sounding(rj, lci,
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
                                     "width": 2.},
                             name = "p10 conductivity",
                             showlegend = False))
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p90']),
                             y = D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'black',
                                     "width": 2.},
                             name = "p90 conductivity",
                             showlegend = False))
    fig.add_trace(go.Scatter(x = np.log10(D['cond_p50']),
                             y = D['depth_cells'],
                             mode = 'lines',
                             line = {"color": 'gray',
                                     "width": 2.,
                                     'dash': 'dash'},
                             name = "p50 conductivity",
                             showlegend = False))

    lci_expanded, depth_expanded = plots.profile2layer_plot(D['lci_cond'], D['lci_depth_top'])

    fig.add_trace(go.Scatter(x=np.log10(lci_expanded),
                             y= depth_expanded,
                             mode='lines',
                             line={"color": 'pink',
                                   "width": 2.,
                                   'dash': 'dash'},
                             name="lci",
                             showlegend=False))


    return fig

def flightline_map(line, vmin, vmax, layer):

    fig = go.Figure()

    cond_grid = np.log10(lci.layer_grids['Layer_{}'.format(layer)]['conductivity'])

    x1, x2, y1, y2 = lci.layer_grids['bounds']
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
                                 #hovertext = ['Line number = ' + str(lineNo)],
                                 line = {"color": c,
                                         "width": 2.},
                                 name = str(lineNo)))

    xmin, xmax = np.min(rj.data['easting'][:]) - 500., np.max(rj.data['easting'][:]) + 500.
    ymin, ymax = np.min(rj.data['northing'][:]) - 500., np.max(rj.data['northing'][:]) + 500.

    fig.update_layout(yaxis=dict(range=[ymin, ymax]),
                      xaxis=dict(range=[xmin, xmax]))
    fig['data'][0]['showscale'] = False
    return fig


section_kwargs = {'colourbar_label': 'Conductivity (S/m)',
                  'vmin': 0.01,
                  'vmax': 1.,
                  'cmap': 'jet'}

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

model = initialise_strat_model(name = modelName)

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
                                    min=0.001, max=10, value = 0.01)],
                         className = 'row'),
                         html.Div(["Conductivity plotting maximum: ", dcc.Input(
                                    id="vmax", type="number",
                                    min=0.001, max=10, value = 1.0)],
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
    html.Div(
        html.Div(
            dcc.Graph(
                id='section_plot',
                figure = {}),
        )
    ),
    html.Div([html.Div(
        html.Div([
            html.Button('Update section', id='update', n_clicks=1),
            dash_table.DataTable(id='interp_table',
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                fixed_columns={ 'headers': True},#, 'data': 1 },
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
                                          'maxWidth':  '500px',
                                          'overflowX': 'scroll'}),

        ],
            className = 'row')
        , className = "five columns"),
        html.Div(html.Div(id='poly_line_plot'), className = "four columns"),
        html.Div(html.Div(id='pmap'), className = "three columns"), ], className = 'row'

             ),
    html.Div(id = 'output')

])

@app.callback(
    [Output('interp_table', 'data'),
    Output('interp_table', 'columns')],
    [Input("line_dropdown", 'value'),
     Input("update", 'n_clicks')])
def update_data_table(value, nclicks):
    if nclicks >0:
        df_ss = subset_df_by_line(model.interpreted_points,
                                  line = value)
        return df_ss.to_dict('records'), [{"name": i, "id": i} for i in df_ss.columns]

@app.callback([Output("surface_dropdown", 'value'),
               Output("surface_dropdown", 'options')],
              [Input('template-upload', 'contents')],
              [State('template-upload', 'filename')])

def update_output(contents, filename):
    if contents is None:
        return model.surfaces[0], model.surface_options
    df = parse_contents(contents, filename)

    valid = True#check_validity(df)
    ##TODO write a file validity function

    if valid:
        # If interpreted points is empty
        if len(model.interpreted_points) < 1:
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

        return model.template.values[0], model.surface_options
    else:
        raise Exception("Input file is not valid")

@app.callback(
    Output('export_message', 'children'),
    [Input("export", 'n_clicks'),
    Input("surface_dropdown", 'value')],
    State('export-path', 'value'))
def export_data_table(nclicks, surfaceName, value):

    if np.logical_and(nclicks > 0, value is not None):
        if os.path.exists(os.path.dirname(value)):
            model.interpreted_points.reset_index(drop=True).to_csv(value)
            return "Successfully exported to " + value
        else:
            return value + " is an invalid file path."

@app.callback(
    Output('section_plot', "figure"),
    [Input("line_dropdown", 'value'),
     Input("section_dropdown", 'value'),
     Input("surface_dropdown", 'value'),
     Input('interp_table', "derived_virtual_data"),
     Input('interp_table', "derived_virtual_selected_rows"),
     Input('vmin', 'value'),
     Input('vmax', 'value')
     ])
def update_section(line, section_plot, surfaceName, rows, derived_virtual_selected_rows, vmin, vmax):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = model.interpreted_points if rows is None else pd.DataFrame(rows)

    select_mask = np.where([True if i in derived_virtual_selected_rows else False
              for i in range(len(dff))])[0]

    section_kwargs['section_plot'] = section_plot

    fig = dash_section(line, dff, select_mask, vmin, vmax, cmap = 'jet')

    return fig

@app.callback(
    [Output('poly_line_plot', 'children')],
    [Input("line_dropdown", 'value'),
     Input('vmin', 'value'),
     Input('vmax', 'value'),
     Input('layerGrid', 'value')])
def update_polyline_plot(line, vmin, vmax, layer):
    fig = flightline_map(line, vmin, vmax, layer)
    return [
        dcc.Graph(
            id='polylines',
            figure=fig
            ),
    ]

@app.callback(
    Output('click-data', 'children'),
    [Input('section_plot', 'clickData'),
     Input("line_dropdown", 'value'),
     Input("surface_dropdown", 'value')])
def update_interp_table(clickData, line, surfaceName):
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    if trig_id == 'section_plot.clickData':

        if clickData['points'][0]['curveNumber'] == 1:
            surface = getattr(model, surfaceName)
            eventxdata, eventydata = clickData['points'][0]['x'], clickData['points'][0]['y']
            min_idx = np.argmin(np.abs(lci.section_data[line]['grid_distances'] - eventxdata))

            easting = lci.section_data[line]['easting'][min_idx]
            northing = lci.section_data[line]['northing'][min_idx]
            elevation = lci.section_data[line]['elevation'][min_idx]
            depth =  elevation - eventydata
            fid = xy2fid(easting,northing, lci)

            # append to the surface object interpreted points
            interp = {'fiducial': fid,
                      'inversion_name': "lci",
                      'X': np.round(easting,0),
                      'Y': np.round(northing,0),
                      'DEPTH': np.round(depth,0),
                      'ELEVATION': eventydata,
                      'DEM': elevation,
                      'UNCERTAINTY': np.nan, # TODO implement
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
            df = pd.DataFrame(interp, index = [0])

            model.interpreted_points = model.interpreted_points.append(df)#, verify_integrity = True)

            return "Last interpretation was ", eventxdata, " along line and ", eventydata, " mAHD"

@app.callback(
    Output('pmap', 'children'),
    Input('section_plot', 'clickData'))
def update_pmap_plot(clickData):
    if clickData is not None:
        if clickData['points'][0]['curveNumber'] == 3:
            point_idx = clickData['points'][0]['pointIndex']
            fig = dash_pmap_plot(point_idx)
            return [
                    dcc.Graph(
                        id='pmap_plot',
                        figure=fig
                        ),
                    ]

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


app.run_server(debug = True)