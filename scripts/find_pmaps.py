# Find pmaps in the structure and copy them into combined

import os, sys, shutil

indir = sys.argv[1]

new_dir = os.path.join(indir, 'combined\pmaps')

if not os.path.exists(new_dir):
    os.mkdir(new_dir)


# Iterate
for dirpath, dirnames, filenames in os.walk(os.path.join(indir, ".")):
    for filename in [f for f in filenames if f.endswith('.pmap')]:
        shutil.copyfile(os.path.join(dirpath, filename),
                        os.path.join(new_dir, filename))
            