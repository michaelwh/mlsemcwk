import pylab

from matplotlib.figure import Figure

import semdatautil
import dataviews

import svmutil

import pprint


from matplotlib.figure import Figure

import viewdata

import math

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import cPickle as pickle

import numpy as np
import matplotlib

from collections import defaultdict

import PyML
import PyML.classifiers
import PyML.classifiers.multi

def make_confusion(actuallabels, predictedlabels):
    ddi = lambda : defaultdict(int)
    confusion = defaultdict(ddi)
    for actual, predicted in zip(actuallabels, predictedlabels):
        # make sure the labels are integers so they can be compared correctly
        actual = int(actual)
        predicted = int(predicted)
        confusion[actual][predicted] += 1
    return confusion

def write_confusion(confusion, file):
    for i in range(10):
        file.write("& \\textbf{{{0}}} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} & {9} & {10} \\\\\n".format(i, confusion[i][0], confusion[i][1], confusion[i][2], confusion[i][3], confusion[i][4], confusion[i][5], confusion[i][6], confusion[i][7], confusion[i][8], confusion[i][9]))


if __name__ == "__main__":
    
    rf = open('results', 'w')
    
    valid_labels = [0,1,2,3,4,5,6,7,8,9]
    
    datarows, o_datalabels = semdatautil.get_sem_data('semeion.data')
    
    print len(datarows)
    
    # convert the data to numeric labels
    datalabels = [int(label) for label in o_datalabels]
    
    
    # SPLIITING IN HALF WILL WORK SINCE IT IS NOT JUST 0 1 2 3 4 5 6 7 8 9 IT IS 0 1 ... 9 0 1 2 ... 9 0 1 2 ETC
    trainingdata = datarows[:len(datarows)/3] 
    traininglabels = datalabels[:len(datalabels)/3]
    
    paramseldata = datarows[len(datarows)/3:(len(datarows)/3)*2] 
    paramsellabels = datalabels[len(datalabels)/3:(len(datalabels)/3)*2]
        
    testingdata = datarows[(len(datarows)/3)*2:] 
    testinglabels = datalabels[(len(datalabels)/3)*2:]
    
    # we will do one against all - ie we will construct one classifier for each data class and where that data class is one class while all other data classes are another class
    

    # now we train our SVM models
    #models = multiclass_train(valid_labels, traininglabels, trainingdata)

    #multiclass_predictions = multiclass_predict(valid_labels, models, testinglabels, testingdata)
    
    #print prediction_accuracy(multiclass_predictions, traininglabels)
    
    
    #oaa_labels = relabel_one_against_all(traininglabels, i)
    #===========================================================================
    # prob = svmutil.svm_problem(traininglabels, trainingdata)
    # param = svmutil.svm_parameter('-t 2')
    # model = svmutil.svm_train(prob, param)
    # predicted_labels, accuracy, decision_vals = svmutil.svm_predict(testinglabels, testingdata, model)
    # 
    # print prediction_accuracy(predicted_labels, testinglabels)
    #===========================================================================
    
    ttlabels = []
    ttlabels.extend(traininglabels)
    ttlabels.extend(testinglabels)
    ttdata = []
    ttdata.extend(trainingdata)
    ttdata.extend(testingdata)

    
    prob = svmutil.svm_problem(traininglabels, trainingdata)
    #prob = svmutil.svm_problem(traininglabels.extend(testinglabels), trainingdata.extend(testingdata))
 
    param = svmutil.svm_parameter('-s 0 -t 2 -c {0} -g {1}'.format(np.exp(0.5), np.exp(-3.56)))
    #accuracy = svmutil.svm_train(prob, param)
    model = svmutil.svm_train(prob, param)
    predicted_labels, accuracy, decision_vals = svmutil.svm_predict(testinglabels, testingdata, model)
    
    
    
    # constructing a confusion matrix


    svm_confusion = make_confusion(testinglabels, predicted_labels)
    print svm_confusion
    
    svm_conf_file = open('svm_conf', 'w')
    write_confusion(svm_confusion, svm_conf_file)
    svm_conf_file.close()
    
    # print latex table to file

    trainingpymldata = PyML.VectorDataSet(trainingdata, L=traininglabels)
    testingpymldata = PyML.VectorDataSet(testingdata, L=testinglabels)
    
    #knn = PyML.classifiers.multi.OneAgainstOne(PyML.KNN(kernel))
    #===========================================================================
    # knn = PyML.KNN(5)
    # knn.train(trainingpymldata)
    # knn_results = knn.test(testingpymldata)
    # print knn_results.getSuccessRate()
    # 
    #===========================================================================
    import myknn
    
    print "knn"
    knn_predicted_labels = []
    correct = 0
    for i in range(len(testingdata)):
        plab = myknn.k_nearest_get_class_color(testingdata[i], trainingdata, traininglabels, k=6)
        knn_predicted_labels.append(plab)
        if plab == testinglabels[i]:
            correct += 1
            print i, "correct"
        else:
            print i, "wrong"
    knn_class_acc = float(correct)/float(len(testingdata))
    print "KNN Classification Accuracy = ", knn_class_acc
    
    knn_confusion = make_confusion(testinglabels, knn_predicted_labels)
    print knn_confusion
    
    knn_conf_file = open('knn_conf', 'w')
    write_confusion(knn_confusion, knn_conf_file)
    knn_conf_file.close()
    
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax = fig.add_subplot(111)
    #ax.plot_surface(caxis, gaxis, accs, cmap=matplotlib.cm.jet, alpha=0.3, rstride=1, cstride=1)
    #ax.set_zlabel('Classification Accuracy (%)')
    
    #ax.set_xlabel('C')
    #ax.set_ylabel('Gamma')

    
    # Automatic selection of levels works; setting the
    # log locator tells contourf to use a log scale:
    #cs = plt.contour(caxis, gaxis, accs)
    #plt.clabel(cs, inline=1, fontsize=10)    
    #cbar = plt.colorbar()
    #cbar.set_label('Classification Accuracy (%)')
    
    
    
    

    
    
    
    #plt.show()


    
    
    
    
    #fig = Figure()#figsize=(10,10))#, dpi=100)
    #ax = fig.add_subplot(111)
    #ax.set_title('Data')
    
    #dataviewer = viewdata.DataViewer(fig, ax, testingdata, labels=predicted_labels)
    
    # now we predict our data
    
    rf.close()
    
   
    
    
    
    

         
    