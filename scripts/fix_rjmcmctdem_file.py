# Fix fixed-width delimiter errors in rjmcmctdem.dat
# This is caused by long run times giving sample times of 
# >10**4 seconds
import sys

infile = sys.argv[1]

outfile = infile.replace(".dat", "_fixed.dat")

with open(infile, 'r' ) as inf:
    with open(outfile, 'w') as outf:
        for line in inf:
            new_line = line[:136] + ' ' + line[136:]
            outf.write(new_line)

# Now chenge the .dfn file

infile = infile.replace('.dat', '.dfn')

outfile = infile.replace(".dfn", "_fixed.dfn")

with open(infile, 'r') as inf:
    with open(outfile, 'w') as outf:
        s = inf.read()
        outf.write(s.replace('sampletime : F8.2', 'sampletime : F9.2'))
    