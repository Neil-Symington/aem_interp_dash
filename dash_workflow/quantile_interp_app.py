import os
import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_html
from scipy import stats
import pandas as pd
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import math
import time
from garjmcmctdem_utils import spatial_functions, aem_utils, netcdf_utils
from garjmcmctdem_utils import plotting_functions as plots
from garjmcmctdem_utils.misc_utils import pickle2xarray, xarray2pickle
import yaml
import netCDF4

def cond2gradient(conductivity):
    gradient = np.zeros(shape = conductivity.shape, dtype = conductivity.dtype)
    gradient [1:] = np.log10(conductivity[1:]) - np.log10(conductivity[:-1])
    return gradient


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


    return {'conductivity_pdf': cond_pdf, "change_point_pdf": cp_pdf, "conductivity_extent": extent,
            'cond_p10': p10, 'cond_p50': p50, 'cond_p90': p90, 'depth_cells': depth_cells,
            'cond_cells': cond_cells, 'det_cond': det_cond, 'det_depth_top': det_depth_top, 'det_doi': det_doi,
            'line': line, 'northing': northing, 'easting': easting, 'fiducial': fiducial,
            'elevation': elevation, 'n_histogram_samples': n_hist_samples}

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

def dash_pmap_plot(fiducial, interp_depth = 0.):
    point_index = np.argmin(np.abs(fiducial - rj.data['fiducial'][:]))
    colours = {'p10': 'blue', 'p50': 'orange', 'p90': 'green'}
    # Extract the data from the netcdf data
    D = extract_rj_sounding(rj, det, point_index)
    pmap = D['conductivity_pdf']
    x1,x2,y1,y2 = D['conductivity_extent']
    n_depth_cells, n_cond_cells  = pmap.shape

    x = 10**np.linspace(x1,x2, n_cond_cells)
    y = np.linspace(y2,y1, n_depth_cells)

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        horizontal_spacing=0.01,
                        column_widths=[0.5, 0.25, 0.25])

    fig.add_trace(go.Heatmap(z = pmap,
                            zmin = 0,
                            zmax = np.max(pmap),
                            x = x,
                            y = y,
                            colorscale ='greys',
                            colorbar=dict(
                                 title="probability",
                             )),
                             row = 1, col = 1)

    for item in ['p10', 'p50', 'p90']:
        cond_expanded, depth_expanded = plots.profile2layer_plot(D["_".join(['cond', item])],y)
        #  PLot the median, and percentile plots
        fig.add_trace(go.Scatter(x = cond_expanded,
                             y = depth_expanded,
                             mode = 'lines',
                             line = {"color": colours[item],
                                     "width": 1.},
                             name = "{} conductivity".format(item),
                             showlegend = False,
                             hoverinfo = None,),
                  row = 1, col=1)
    #fig.add_trace(go.Scatter(x = D['cond_p50'],
    #                         y = y,
    #                         mode = 'lines',
    #                         line = {"color": colours['p50'],
    #                                 "width": 1.},
    #                         name = "p50 conductivity",
    #                         showlegend = False,
    #                         hoverinfo = None,),
    #              row = 1, col=1)
    #fig.add_trace(go.Scatter(x = D['cond_p90'],
    #                         y = y,
    #                         mode = 'lines',
    #                         line = {"color": colours['p90'],
    #                                 "width": 1.5,
    #                                 'dash': 'dash'},
    #                         name = "p90 conductivity",
    #                         showlegend = False,
    #                         hoverinfo = None,),
    #              row = 1, col=1)

    #det_expanded, depth_expanded = plots.profile2layer_plot(D['det_cond'], D['det_depth_top'])

    # fig.add_trace(go.Scatter(x=det_expanded,
    #                          y= depth_expanded,
    #                          mode='lines',
    #                          line={"color": 'pink',
    #                                "width": 1.5,
    #                                'dash': 'dash'},
    #                          name=det.name,
    #                          showlegend=False),
    #               row = 1, col=1)
    # plot the gradient of the conductivity

    for item in ["p10", "p50", "p90"]:
        filtered = medfilt(D["_".join(['cond', item])], 7)

        gradient = cond2gradient(filtered)

        fig.add_trace(go.Scatter(x=gradient,
                                 y=y,
                                 mode='lines',
                                 line={"color": colours[item],
                                       "width": 2.},
                                 name=item,
                                 showlegend=False, hoverinfo = None),
                  row=1, col=2)
    # plot existing interpretation

    row = df_output_interp[df_output_interp['fiducial'] == fiducial]

    for item in ["p10", "p50", "p90"]:
        value = row[item].values[0]
        if not np.isnan(value):
            fig.add_trace(go.Scatter(x=[np.min(D["cond_p10"]), np.max(D["cond_p90"])],
                                     y=[value, value],
                                     mode='lines',
                                     line={"color": colours[item],
                                           "width": 0.5},
                                     name="current_interpretation",
                                     showlegend=False,  hoverinfo = None),
                          row=1, col=1)



    fig.add_trace(go.Scatter(x=D['change_point_pdf'],
                             y= y,
                             mode='lines',
                             line={"color": 'red',
                                   "width": 2.},
                             name="cpp",
                             showlegend=False, hoverinfo = None),
                  row = 1, col=3)

    fig.add_trace(go.Scatter(x=[0, np.max(D['change_point_pdf'])],
                             y=[interp_depth, interp_depth],
                             mode='lines',
                             line={"color": 'grey',
                                   "width": 2.},
                             name="interpreted_point",
                             showlegend=False),
                  row=1, col=3)

    fig.update_layout(
        autosize=False,
        height=800)

    fig.update_layout(xaxis=dict(scaleanchor = 'y',
                                 scaleratio = 100.))
    fig.update_xaxes(type="log", title_text="Conductivity (S/m)",
                     row = 1, col = 1)
    fig.update_yaxes(autorange='reversed', row=1, col=1)

    return fig

def find_trigger():
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    return trig_id

root = "/home/nsymington/Documents/GA/dash_data_Surat"
yaml_file = "interpretation_config.yaml"
settings = yaml.safe_load(open(yaml_file))

interp_settings, model_settings, AEM_settings, det_inv_settings, stochastic_inv_settings, section_settings,\
borehole_settings, crs = settings.values()

det = aem_utils.AEM_inversion(name = det_inv_settings['inversion_name'],
                              inversion_type = 'deterministic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, det_inv_settings['nc_path'])))

rj = aem_utils.AEM_inversion(name = stochastic_inv_settings['inversion_name'],
                              inversion_type = 'stochastic',
                              netcdf_dataset = netCDF4.Dataset(os.path.join(root, stochastic_inv_settings['nc_path'])))

df_interpreted_points = pd.read_csv(model_settings['interpFile'])

# now create a copy for saving new interpretations

#df_output_interp = df_interpreted_points[['fiducial', 'DEPTH']].copy().rename(columns = {"DEPTH": "p50"})
df_output_interp = pd.read_csv('/home/nsymington/Documents/GA/GAB/Injune/quantile_interp.csv')[["fiducial", "p50", "p10", "p90"]]

#print(df_output_interp)

#df_output_interp['p10'] = np.nan
#df_output_interp['p90'] = np.nan

print(df_output_interp)

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

app.layout = html.Div([
    html.Div([
        html.Div(["Select quantile to interpret: ",
                             dcc.Dropdown(id = "quantile_dropdown",
                            options=[{'label': 'p10', 'value': 'p10'},
                                     {'label': 'p50', 'value': 'p50'},
                                     {'label': 'p90', 'value': 'p90'}],
                            value=('p50'))
                ],
                 className = 'three columns'),
        html.Div([
            html.Div(html.Button('Export results', id='export', n_clicks=0)),
                     html.Div(dcc.Input(id='export-path', type='text',
                                        placeholder = 'Input valid output path')),
                     html.Div(id='export_message')
                             ],className = "three columns")
    ],
        className = "row"),
    html.Div([
        html.Div(
                dcc.Graph(id = 'pmap_plot', style = dict(height = 800)),
                            className = 'eight columns'),
        html.Div(dash_table.DataTable(id='master_table',
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                columns = [{"name": i, "id": i} for i in df_interpreted_points.columns],
                                data=df_interpreted_points.to_dict('records'),
                                fixed_columns={ 'headers': True},
                                sort_action="native",
                                sort_mode="multi",
                                row_selectable="single",
                                row_deletable=False,
                                selected_columns=[],
                                selected_rows=[0],
                                style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                             'height': '40px'},
                                style_cell={
                                             'backgroundColor': 'rgb(50, 50, 50)',
                                             'color': 'white',
                                             'minHeight': '50px',
                                             'minWidth': '0px', 'maxWidth': '80px',
                                             'whiteSpace': 'normal',
                                             'font-size': '10px'
                                         },
                                style_table={
                                          'maxHeight': '800px',
                                          'overflowY': 'scroll',
                                          'maxWidth':  '90%',
                                          'overflowX': 'scroll'}),
                 className = 'four columns')
    ],
        className = "row"),
    html.Div(id='interp_message')

])

# Render pmap plot
@app.callback(Output('pmap_plot', 'figure'),
              Input('master_table', "derived_virtual_selected_rows"))
def update_plot(selected_rows):
    selected_rows = selected_rows or [0]
    row = df_interpreted_points.iloc[selected_rows]
    interp_depth = float(row['DEPTH'].values[0])
    fiducial = float(row['fiducial'].values[0])

    return dash_pmap_plot(fiducial, interp_depth)

@app.callback(Output('interp_message', 'children'),
              Input('pmap_plot', 'clickData'),
              [State("quantile_dropdown", "value"),
              State('master_table', "derived_virtual_selected_rows")])
def click_update(clickData, quantile, selected_rows):
    clickData = clickData or None
    if clickData is None:
        return ""
    eventydata = clickData['points'][0]['y']
    row = df_interpreted_points.iloc[selected_rows]
    # update dataframe
    df_output_interp.at[row.index, quantile] = eventydata
    return " ".join([quantile, "interpreted depth is ",
                     str(np.round(eventydata, 1)), "m"])

@app.callback(
    Output('export_message', 'children'),
    [Input("export", 'n_clicks')],
    [State('export-path', 'value')])
def export_data_table(nclicks, value):

    if np.logical_and(nclicks > 0, value is not None):
        if os.path.exists(os.path.dirname(value)):
            df_merged = df_output_interp.merge(df_interpreted_points, on='fiducial')
            df_merged.to_csv(value, index=False)
            return "Successfully exported to " + value
        else:
            return value + " is an invalid file path."

if __name__ == '__main__':
    app.run_server(debug=False)