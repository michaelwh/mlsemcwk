"""
make a scatter plot with varying color and size arguments
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import random
import math

def k_nearest_get_class_color(point, points, k=5):
    dists = []
    for p in points:
        dists.append((math.sqrt(math.pow(p['x'] - point['x'],2) + math.pow(p['y'] - point['y'],2)), p))
    dists.sort()
    # find class based on closest k neighbors
    i = 0
    noa = 0
    nob = 0
    for dist, p in dists:
        if p['class'] == 'a':
            noa += 1
        else:
            nob += 1
        if i >= k:
            break    
        i += 1

    if noa > nob:
        return 'b'
    else:
        return 'r'


points = []
for i in range(100):
    point = {}    
    point['x'] = random.uniform(0,2)
    point['y'] = random.uniform(0,2)
    if point['x'] > 1:
        if random.uniform(0,1) > 0.9:
            point['class'] = 'a'
        else:
            point['class'] = 'b'
    else:
        if random.uniform(0,1) > 0.9:
            point['class'] = 'b'
        else:
            point['class'] = 'a'
    points.append(point)

fig = plt.figure()
ax = fig.add_subplot(111)
for point in points:
    if point['class'] == 'a':

        ax.scatter(point['x'], point['y'], marker='s', c=k_nearest_get_class_color(point, points))
    else:
        ax.scatter(point['x'], point['y'], marker='^', c=k_nearest_get_class_color(point, points))

plt.show()


