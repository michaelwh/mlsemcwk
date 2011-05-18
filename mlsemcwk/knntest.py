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
from collections import defaultdict
import operator


def k_nearest_get_class_color(point, points, points_labels, k=5):
    dists = []
    for pin in range(len(points)):
        dists.append((math.sqrt(math.sum((math.pow(points[pin][i] - point[i],2) for i in range(len(point))))), points_labels[pin]))
    dists.sort()
    # find class based on closest k neighbors
    i = 0
    classcount = defaultdict(int)
    for dist, p_label in dists:
        classcount[p_label] += 1
        if i >= k:
            break    
        i += 1

    return max(classcount.iteritems(), key=operator.itemgetter(1))[0]


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


