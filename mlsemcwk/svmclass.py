import PyML
import PyML.classifiers
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

svm_results = svm.test(testingpymldata)

print svm_results

# now attempt KNN
knn_num_neighbors = 3
knn = PyML.classifiers.KNN(knn_num_neighbors)
knn.train(trainingpymldata)
knn_results = knn.test(testingpymldata)
print knn_results

# now attempt Ridge Regression
rr_regularization_param = 1
rr = PyML.classifiers.RidgeRegression(rr_regularization_param)
rr.train(trainingpymldata)
rr_results = rr.test(testingpymldata)
print rr_results

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
print "TRAINING TIME: " + str(svm.getTrainingTime())

dataviewer = dataviews.DataViewer(fig, ax, testingdatarows, actuallabels=svm_results.getGivenLabels(), predictedlabels=svm_results.getPredictedLabels())
#dataviewer = dataviews.DataViewer(fig, ax, [mynumrow], predictedlabels=mynumresults.getPredictedLabels())


# TEST ONE AGAINST MANY AND ONE AGAINST ONE
# TEST KNN AND RIGDE REGRESSION
# TEST TIME OF ALL CLASSIFIERS
# TEST CLASSIFIERS WITH DIFFERING PARAMETERS
#   -   EG SVM WITH POLYNOMIAL KERNEL WITH DIFFERENT ORDER POLYNOMIAL
#   -   MINIMISE OVER THIS