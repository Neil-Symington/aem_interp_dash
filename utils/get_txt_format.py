import os
import re

infile = r"C:\Users\u77932\Documents\EFTF2\SW\data\existing\AEM\110284_Data_Package\AEM\EM_located_data\AUS_10013_Musgrave_EM\AUS_10013_Musgrave_EM.dat"

lines = ""
with open(infile, 'r') as f:
    for i in range(2000):
        line = f.readline()
        lines+= line

list = re.split(r'(\s+)', line)[1:-1]

format = []
for i in range(int(len(list)/2)):
    s = list[2*i] + list[2*i+1]
    #print(s)
    if '.' in s:
        fmt = "".join(["F", str(len(s)), ".", str(len(s.split('.')[1]))])
    else:
        fmt = "".join(["I",str(len(s))])
    format.append(fmt)

with open(r"C:\temp\someline.dat", 'w') as f:
    f.write(lines)

print(line)
print(format)