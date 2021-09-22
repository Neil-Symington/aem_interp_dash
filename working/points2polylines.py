import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

infile = r"\\prod.lan\active\proj\futurex\OMP\Data\Original\Geophysics\AEM\2016_Musgraves_SkyTEM\Shapefiles\FPath\Shape\AUS_10013_Musgrave_FPath.shp"

gdf = gpd.read_file(infile)

geom = gdf['geometry'].values

easting = np.array([v.x for v in geom])
northing = np.array([v.y for v in geom])

lines = gdf['Line'].values

flight_lines = {}

crs = "EPSG:28352"

for i, line in enumerate(np.unique(lines)):
    mask = np.where(lines == line)[0]
    # now get the easting and northing and sort
    x = easting[mask]
    y = northing[mask]

    # add the polyline to the attribute
    flight_lines[line] = LineString(np.column_stack((x, y)))

flightlines = gpd.GeoDataFrame(data = {'lineNumber': flight_lines.keys(),
                                             'geometry': flight_lines.values()},
                                            geometry= 'geometry', crs = crs)

flightlines.to_file(r"C:\Users\u77932\Documents\MORPH\data\AEM\2016_SkyTEM\Shapefiles\skytem_2016_flightlines.shp")