import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import random
import math

fig = plt.figure()
ax = fig.add_subplot(111)

df = open('semeion.data', 'r')
lineimg = df.readline().rsplit(' ')
img = []
# extract 16 lines
for line in range(16):
    # extract 16 pixels from each line (and convert each one to floats)
    img.append([float(pix) for pix in lineimg[line*16:(line*16 + 16)]])

ax.imshow(img, cmap=plt.gray())

plt.show()


