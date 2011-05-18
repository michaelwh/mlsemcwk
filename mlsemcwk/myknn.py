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
        dists.append((math.sqrt(math.fsum((math.pow(float(points[pin][i]) - float(point[i]),2) for i in range(len(point))))), points_labels[pin]))
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
    #return classcount
