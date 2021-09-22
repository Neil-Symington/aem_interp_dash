import imageio
import os

fnames = [r"C:\Users\u77932\Documents\GAB\working\plots\swarm_plots\{}_swarm_plot.png".format(i) for i in range(1,101)]

with imageio.get_writer(r"C:\Users\u77932\Documents\GAB\working\plots\swarm.gif", mode='I') as writer:
    for filename in fnames:
        image = imageio.imread(filename)
        writer.append_data(image)