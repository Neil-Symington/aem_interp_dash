# A script for chekcing misc column statistics for .dat files

import pandas as pd
import os, gc
import numpy as np

infile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\EM_located_data\AUS_10013_Musgrave_EM\AUS_10013_Musgrave_EM_corrected.dat"

df = pd.read_fwf(infile)

bad_lines = np.where(df.iloc[:,3].values > 912999)


null = -9999999.99999

# assuming the first row is correct
null_cols = np.where(df.iloc[0] == null)[0]

# we will append the bad rows to a list
bad_rows = []

for col in null_cols:
    if not np.all(df.iloc[:, col].unique() == np.array([null])):
        bad_rows.append(np.where(df.iloc[:, col] != null)[0])

bad_rows = np.where(df.iloc[:,3] == 913001)[0]

bad_rows = bad_rows + 1

bad_rows = []

bad_rows = np.where(df.iloc[:,35] > 999.)[0]

bad_rows = bad_rows + 1

#df = None


outfile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\EM_located_data\AUS_10013_Musgrave_EM\AUS_10013_Musgrave_EM_high_corrected_again.dat"

with open(infile, 'r') as f:
    with open(outfile, 'w') as outf:
        for i, line in enumerate(f):
            if not i in bad_rows:
                outf.write(line)
            else:
                print(i)