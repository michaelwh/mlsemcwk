import PyML
import PyML.classifiers.multi

import pylab

from matplotlib.figure import Figure

import semdatautil
import dataviews



datarows, labels = semdatautil.get_sem_data('semeion.data')


# SPLIITING IN HALF WILL WORK SINCE IT IS NOT JUST 0 1 2 3 4 5 6 7 8 9 IT IS 0 1 ... 9 0 1 2 ... 9 0 1 2 ETC
trainingdatarows = datarows[len(datarows)/2:] 
traininglabels = labels[len(labels)/2:]

testingdatarows = datarows[:len(datarows)/2] 
testinglabels = labels[:len(labels)/2:]

trainingpymldata = PyML.VectorDataSet(trainingdatarows, L=traininglabels)
testingpymldata = PyML.VectorDataSet(testingdatarows, L=testinglabels)

#kernel = PyML.ker.Polynomial()
kernel = PyML.ker.Linear()
#svm = PyML.classifiers.multi.OneAgainstRest(PyML.SVM(kernel))
svm = PyML.classifiers.multi.OneAgainstOne(PyML.SVM(kernel))

svm.train(trainingpymldata)
results = svm.test(testingpymldata)

print results


# now classify our own image
mynumimg = pylab.mean(pylab.imread('mynum.png'), 2)

mynumrow = []
for mynumline in mynumimg:
    for mynumpix in mynumline:
        mynumrow.append(float(mynumpix))
        
mynumpymldata = PyML.VectorDataSet([mynumrow])
mynumresults = svm.test(mynumpymldata)

print "MY NUM RESULT: " + str(mynumresults.getPredictedLabels()[0])

# now display data

fig = Figure()#figsize=(10,10))#, dpi=100)
ax = fig.add_subplot(111)
ax.set_title('Data')

dataviewer = dataviews.DataViewer(fig, ax, testingdatarows, actuallabels=results.getGivenLabels(), predictedlabels=results.getPredictedLabels())
#dataviewer = dataviews.DataViewer(fig, ax, [mynumrow], predictedlabels=mynumresults.getPredictedLabels())
