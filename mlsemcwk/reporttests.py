import PyML
import PyML.classifiers
import PyML.classifiers.multi

import pylab

from matplotlib.figure import Figure
from matplotlib import pyplot

import numpy as np

import semdatautil
import dataviews
import matplotlib


datarows, labels = semdatautil.get_sem_data('semeion.data')


# SPLIITING IN HALF WILL WORK SINCE IT IS NOT JUST 0 1 2 3 4 5 6 7 8 9 IT IS 0 1 ... 9 0 1 2 ... 9 0 1 2 ETC
trainingdatarows = datarows[len(datarows)/2:] 
traininglabels = labels[len(labels)/2:]

testingdatarows = datarows[:len(datarows)/2] 
testinglabels = labels[:len(labels)/2:]

trainingpymldata = PyML.VectorDataSet(trainingdatarows, L=traininglabels)
testingpymldata = PyML.VectorDataSet(testingdatarows, L=testinglabels)

resf = open('results', 'w')

fig = pyplot.figure()#figsize=(10,10))#, dpi=100)

# --- Gaussian kernel width
#resf.write("Gaussian Kernel Width Results\n===\n")
gauss_gamma = []
gauss_bsuccrate = []
for gamma in np.logspace(0,3,10):
    kernel = PyML.ker.Gaussian(gamma)
    svm = PyML.classifiers.multi.OneAgainstOne(PyML.SVM(kernel))

    svm.train(trainingpymldata)
    
    svm_results = svm.test(testingpymldata)
    
    gauss_gamma.append(gamma)
    gauss_bsuccrate.append(svm_results.balancedSuccessRate)

print gauss_gamma    
print gauss_bsuccrate


gauss_ax = fig.add_subplot(111)
gauss_ax.plot(gauss_gamma, gauss_bsuccrate, 'o-')
gauss_ax.set_title('Gaussian Kernel')
gauss_ax.set_xscale('log')
gauss_ax.set_xlabel('Gaussian Width')
gauss_ax.set_ylabel('Success Rate')

fig.savefig('gauss_gamma.png')
pyplot.show()


#resf.close()

# TEST ONE AGAINST MANY AND ONE AGAINST ONE
# TEST KNN AND RIGDE REGRESSION
# TEST TIME OF ALL CLASSIFIERS
# TEST CLASSIFIERS WITH DIFFERING PARAMETERS
#   -   EG SVM WITH POLYNOMIAL KERNEL WITH DIFFERENT ORDER POLYNOMIAL
#   -   MINIMISE OVER THIS