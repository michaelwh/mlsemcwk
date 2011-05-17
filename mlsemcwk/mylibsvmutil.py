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

def relabel_one_against_all(labels, classindex):
    """Relabels the 'one' class to 1 and the 'rest' class to -1"""
    outlabels = []
    for i in range(len(labels)):
        if labels[i] == classindex:
            outlabels.append(1)
        else:
            outlabels.append(-1)
    return outlabels

def print_no_of_one_against_all_classes(one_against_all_data):
    for i in range(len(one_against_all_data)):
        num = 0
        for label in one_against_all_data[i]:
            if label == 1:
                num += 1
        print str(i) + ": " + str(num)
        
def multiclass_train(valid_labels, labels, data, svm_parameters=None):
    if svm_parameters == None:
        # make default empty parameters 
        svm_parameters = []
        for i in valid_labels:
            svm_parameters.append(svmutil.svm_parameter())
    models = []
    for i in valid_labels:
        oaa_labels = relabel_one_against_all(labels, i)
        prob = svmutil.svm_problem(oaa_labels, data)
        model = svmutil.svm_train(prob, svm_parameters[i])
        models.append(model)
    return models

def multiclass_predict(valid_labels, multiclass_models, labels, data):
    multiclass_decision_vals = []
    for i in range(len(multiclass_models)):
        oaa_labels = relabel_one_against_all(labels, valid_labels[i])
        predicted_labels, accuracy, decision_vals = svmutil.svm_predict(oaa_labels, data, multiclass_models[i])
        multiclass_decision_vals.append(decision_vals)
        print predicted_labels
        print decision_vals
    # now we have the decision values for every classifier for every datapoint we can make our decision
    # for each datapoint
    multiclass_predicted_labels = []
    for i in range(len(data)):
        # for each datapoint, now we must look at the decision values for each classifier and find the highest 
        # (most positive) as this will correspond to the most probable class for this datapoint
        max_class = valid_labels[0] # for now assume is the first label
        max_value = multiclass_decision_vals[0][i]
        for k in range(1,len(multiclass_models)):
            if multiclass_decision_vals[k][i] > max_value:
                max_class = valid_labels[k]
                max_value = multiclass_decision_vals[k][i]
        multiclass_predicted_labels.append(max_class)
    return multiclass_predicted_labels

def prediction_accuracy(predicted_labels, actual_labels):
    numcorrect = 0
    for i in range(len(predicted_labels)):
        if int(predicted_labels[i]) == int(actual_labels[i]):
            numcorrect += 1
    return numcorrect, len(predicted_labels), float(numcorrect)/float(len(predicted_labels))
    
if __name__ == "__main__":
    
    rf = open('results', 'w')
    
    valid_labels = [0,1,2,3,4,5,6,7,8,9]
    
    datarows, o_datalabels = semdatautil.get_sem_data('semeion.data')
    
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
    
    prob = svmutil.svm_problem(traininglabels, trainingdata)
    
    # now do a grid search for C and gamma
    c_vals = [np.exp(x) for x in np.linspace(0,4,10)]
    gamma_vals = [np.exp(x) for x in np.linspace(-3.53,-3.57,10)]
    #c_vals = np.linspace(0.1,1,3)
    #gamma_vals = np.linspace(0.4,1,3)
    #gamma_vals = [0.05,0.03,0.01]
    #c_vals = [math.pow(2, x) for x in range(100,104, 1)]
    #gamma_vals = [math.pow(2, x) for x in range(100,104,1)]
    #c_vals = [0.1,1]
    #gamma_vals = [0.1,2]
    print "c", c_vals
    print "gamma", gamma_vals
    
    redo = True
    
    if redo == True:
        accs = np.zeros((len(c_vals), len(gamma_vals)))
        for c_index in range(len(c_vals)):
            for gamma_index in range(len(gamma_vals)):
                param = svmutil.svm_parameter('-s 0 -t 2 -c {0} -g {1}'.format(c_vals[c_index], gamma_vals[gamma_index]))
                #accuracy = svmutil.svm_train(prob, param)
                model = svmutil.svm_train(prob, param)
                predicted_labels, accuracy, decision_vals = svmutil.svm_predict(paramsellabels, paramseldata, model)
                accs[gamma_index, c_index] = accuracy[0]
        gr = open("grid_results", 'w')
        pickle.dump(accs, gr)
        gr.close()
    else:
        gr = open("grid_results", 'r')
        accs = pickle.load(gr)
        gr.close()
        
    
    
    caxis, gaxis = np.meshgrid(np.log(c_vals), np.log(gamma_vals))
    print caxis
    print gaxis
    print accs
    
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111)
    #ax.plot_surface(caxis, gaxis, accs, cmap=matplotlib.cm.jet, alpha=0.3, rstride=1, cstride=1)
    #ax.set_zlabel('Classification Accuracy (%)')
    
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')

    
    # Automatic selection of levels works; setting the
    # log locator tells contourf to use a log scale:
    cs = plt.contour(caxis, gaxis, accs)
    plt.clabel(cs, inline=1, fontsize=10)    
    cbar = plt.colorbar()
    cbar.set_label('Classification Accuracy (%)')
    
    plt.show()


    
    
    
    
    #fig = Figure()#figsize=(10,10))#, dpi=100)
    #ax = fig.add_subplot(111)
    #ax.set_title('Data')
    
    #dataviewer = viewdata.DataViewer(fig, ax, testingdata, labels=predicted_labels)
    
    # now we predict our data
    
    rf.close()
    
   
    
    
    
    

         
    