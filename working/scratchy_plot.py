import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

infile = r"E:\GA\EK_salinity_mapping\EK_cond_pmap_top10m_saturated_zone.csv"

df = pd.read_csv(infile)

cond_cols = df.columns[4:]

fig, ax = plt.subplots(1,1, figsize = (2,2))

x = np.linspace(np.log10(0.002), np.log10(3.23), 100)

ax.hist(x = x, weights = df.iloc[616][cond_cols], bins = 100)
ax.set_xlabel("log10 bulk conductivity (S/m)")
ax.set_ylabel('count')

plt.show()

infile = r"C:\Users\u77932\Documents\EastKimberley\salinity_mapping\EK_ec_prediction.csv"

df_ec = pd.read_csv(infile)

fig, ax = plt.subplots(1,1, figsize = (2,2))

ax.hist(df_ec.iloc[616].values, bins = 100)
ax.set_xlabel("log10 EC (S/m)")
ax.set_ylabel('count')

plt.show()

infile = r"E:\GA\EK_salinity_mapping\EK_EC_cond_joint_pdf_pmap_AEGC.csv"

df__ = pd.read_csv(infile)

df__['ML'] = np.nan

for index, row in df__.iterrows():
    amax = np.argmax(row[cond_cols].values)
    df__.at[index,'ML'] = x[amax]


fig, ax = plt.subplots(1,1, figsize = (3,3))

ax.scatter(df__['ML'].values, df__['log_EC_mean'].values)

slope, intercept, r_value, p_value, std_err = stats.linregress(df__['ML'].values,
                                                                df__['log_EC_mean'].values,)

y = slope*np.array([-3,1]) + intercept
ax.plot([-3,1], y, c= 'k', label = "linear regression function")
ax.set_xlabel("log10 bulk conductivity (S/m)")
ax.set_ylabel("log10 EC (S/m)")
ax.legend()
plt.show()