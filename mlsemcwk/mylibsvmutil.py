import pylab

from matplotlib.figure import Figure

import semdatautil
import dataviews

import svmutil



def relabel_one_against_all(labels, classindex):
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

if __name__ == "__main__":
    datarows, o_labels = semdatautil.get_sem_data('semeion.data')
    
    # convert the data to numeric labels
    labels = [int(label) for label in o_labels]
    
    
    # SPLIITING IN HALF WILL WORK SINCE IT IS NOT JUST 0 1 2 3 4 5 6 7 8 9 IT IS 0 1 ... 9 0 1 2 ... 9 0 1 2 ETC
    trainingdata = datarows[len(datarows)/2:] 
    traininglabels = labels[len(labels)/2:]
    
    testingdata = datarows[:len(datarows)/2] 
    testinglabels = labels[:len(labels)/2:]
    
    # we will do one against all - ie we will construct one classifier for each data class and where that data class is one class while all other data classes are another class
    
    oaa_training_labels = []
    oaa_testing_labels = []
    
    for i in range(10):
        oaa_training_labels.append(relabel_one_against_all(traininglabels, i))
        oaa_testing_labels.append(relabel_one_against_all(testinglabels, i))

    
    
    prob1 = svmutil.svm_problem(oaa_training_labels[0], trainingdata)
    param1 = svmutil.svm_parameter()
    m1 = svmutil.svm_train(prob1, param1)
    
    svmutil.svm_predict(oaa_training_labels[0], trainingdata, m1)
    

         
    