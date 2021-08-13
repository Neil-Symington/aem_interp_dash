import pandas as pd
import numpy as np

infile = "/home/nsymington/Documents/GA/GAB/Injune/quantile_interp_updated.csv"

df_interp = pd.read_csv(infile)

# create a new dataframe

df_consol = pd.DataFrame(columns = ["surface", "fiducial", "X", "Y", "DEM", "p50", "p10", "p90", "line"])

fids = df_interp['fiducial'].unique()
boundaries = df_interp['BoundaryNm'].unique()

surfaces = np.unique([s.split("_p")[0] for s in boundaries])

for surface in surfaces:
    for fid in fids:
        d = {"surface": [surface], "fiducial": [fid], "p50": [np.nan], "p10": [np.nan], "p90":[np.nan], "line": [0]}
        for item in ["p10", 'p50', 'p90']:
            boundarynm = "_".join([surface, item])
            df_temp = df_interp[(df_interp['fiducial'] == fid) & (df_interp['BoundaryNm'] == boundarynm)]
            if len(df_temp) > 0:
                d[item] = df_temp["DEPTH"].values
                d['X'] = df_temp['X'].values
                d['Y'] = df_temp['Y'].values
                d['DEM'] = df_temp['DEM'].values
                d['line'] = df_temp['SURVEY_LINE'].values
        df_fid = pd.DataFrame(d).dropna(how = 'any', subset = ['p10','p90', 'p50'])
        if not len(df_fid) == 0:
            df_consol = df_consol.append(pd.DataFrame(d))

df_consol.to_csv("/home/nsymington/Documents/GA/GAB/Injune/quantile_interp_consolidated.csv")

