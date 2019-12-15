# Concatenating rjmcmc.dat files from a directory structure

import os, sys

indir = sys.argv[1]

new_dir = os.path.join(indir, 'combined')

if not os.path.exists(new_dir):
    os.mkdir(new_dir)

outfile = os.path.join(new_dir, 'rjmcmc.dat')

s = ''

# Iterate
for dirpath, dirnames, filenames in os.walk(os.path.join(indir, ".")):
    for filename in [f for f in filenames if f == "rjmcmc.dat"]:
        with open(os.path.join(dirpath, filename), 'r') as infile:
            s += infile.read()
        
with open(outfile, 'w') as f:
    f.write(s)