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

import cPickle as pickle

import scipy.optimize as opt

class ReportTests(object):
    
    def __init__(self):
        datarows, labels = semdatautil.get_sem_data('semeion.data')


        # SPLIITING IN HALF WILL WORK SINCE IT IS NOT JUST 0 1 2 3 4 5 6 7 8 9 IT IS 0 1 ... 9 0 1 2 ... 9 0 1 2 ETC
        self.trainingdatarows = datarows[len(datarows)/2:] 
        self.traininglabels = labels[len(labels)/2:]
        
        self.testingdatarows = datarows[:len(datarows)/2] 
        self.testinglabels = labels[:len(labels)/2:]
        
        self.trainingpymldata = PyML.VectorDataSet(self.trainingdatarows, L=self.traininglabels)
        self.testingpymldata = PyML.VectorDataSet(self.testingdatarows, L=self.testinglabels)
        
        rf = open('results', 'w')
        rf.write("========== RESULTS ==========\n")
        rf.close()

    def do_gauss_gamma_log(self):
        # --- Gaussian kernel width
        #resf.write("Gaussian Kernel Width Results\n===\n")
        gauss_gamma = []
        gauss_bsuccrate = []
        for gamma in np.logspace(0,3,10):
            result = self.gaussian_svm(gamma)
            gauss_gamma.append(gamma)
            gauss_bsuccrate.append(result.successRate)
        
        print gauss_gamma    
        print gauss_bsuccrate
        
        gg_results = (gauss_gamma, gauss_bsuccrate)
        
        ggf = open('gg_results_log', 'w')
        pickle.dump(gg_results, ggf)
        ggf.close()
        
        self.plot_gauss_gamma_log()
        
        
    def plot_gauss_gamma_log(self):
        ggf = open('gg_results_log', 'r')
        gauss_gamma, gauss_bsuccrate = pickle.load(ggf)
        ggf.close()
        fig = pyplot.figure()#figsize=(10,10))#, dpi=100)
        gauss_ax = fig.add_subplot(111)
        gauss_ax.plot(gauss_gamma, gauss_bsuccrate, 'o-')
        gauss_ax.set_title('Gaussian Kernel SVM')
        gauss_ax.set_xscale('log')
        gauss_ax.set_xlabel('Gaussian Width (log scale)')
        gauss_ax.set_ylabel('Success Rate')
        fig.savefig('gauss_gamma_log.png')
        
    def do_gauss_gamma(self):
        # --- Gaussian kernel width
        #resf.write("Gaussian Kernel Width Results\n===\n")
        gauss_gamma = []
        gauss_bsuccrate = []
        for gamma in np.linspace(0,1,10):
            result = self.gaussian_svm(gamma)
            gauss_gamma.append(gamma)
            gauss_bsuccrate.append(result.successRate)
        
        print gauss_gamma    
        print gauss_bsuccrate
        
        gg_results = (gauss_gamma, gauss_bsuccrate)
        
        ggf = open('gg_results', 'w')
        pickle.dump(gg_results, ggf)
        ggf.close()
        
        self.plot_gauss_gamma()
        
        
    def plot_gauss_gamma(self):
        ggf = open('gg_results', 'r')
        gauss_gamma, gauss_bsuccrate = pickle.load(ggf)
        ggf.close()
        fig = pyplot.figure()#figsize=(10,10))#, dpi=100)
        gauss_ax = fig.add_subplot(111)
        gauss_ax.plot(gauss_gamma, gauss_bsuccrate, 'o-')
        gauss_ax.set_title('Gaussian Kernel SVM')
        gauss_ax.set_xlabel('Gaussian Width')
        gauss_ax.set_ylabel('Success Rate')
        fig.savefig('gauss_gamma.png')

    def opt_gauss_gamma(self):
        gammas = []
        results = []
        def gauss_svn_opt(gamma):
            result = -rt.gaussian_svm(gamma)
            gammas.append(gamma)
            results.append(result.successRate)
            return result.successRate
        def minprintcallback(xk):
            print "Minimisation iteration " + str(xk)
        opt_result = opt.fmin(gauss_svn_opt, 0.1, callback=minprintcallback)
        print "=== opt_results ==="
        print opt_result
        print "==================="
        print gammas
        print results
        return (opt_result, gammas, results)

    def gaussian_svm(self, gamma):
        kernel = PyML.ker.Gaussian(np.float64(gamma))
        svm = PyML.classifiers.multi.OneAgainstOne(PyML.SVM(kernel))

        svm.train(self.trainingpymldata)

        svm_results = svm.test(self.testingpymldata)

        return svm_results

    def do_linear_svm_test(self):
        kernel = PyML.ker.Linear()
        svm = PyML.classifiers.multi.OneAgainstOne(PyML.SVM(kernel))

        svm.train(self.trainingpymldata)

        svm_results = svm.test(self.testingpymldata)
        rf = open('results', 'a')
        rf.write("Linear SVM Test:\n\tSuccessRate = " + str(svm_results.successRate) + "\n")
        rf.close()
        
    def do_polynomial_svm_test(self):
        poly_order = []
        poly_bsuccrate = []
        for order in range(1,10):
            result = self.poly_svm(order)
            poly_order.append(order)
            poly_bsuccrate.append(result.successRate)
        
        print poly_order    
        print poly_bsuccrate
        
        poly_results = (poly_order, poly_bsuccrate)
        
        polyf = open('poly_results', 'w')
        pickle.dump(poly_results, polyf)
        polyf.close()
        
        self.plot_poly_order()
        
    def plot_poly_order(self):
        polyf = open('poly_results', 'r')
        poly_order, poly_bsuccrate = pickle.load(polyf)
        polyf.close()
        fig = pyplot.figure()#figsize=(10,10))#, dpi=100)
        gauss_ax = fig.add_subplot(111)
        gauss_ax.plot(poly_order, poly_bsuccrate, 'o-')
        gauss_ax.set_title('Polynomial Kernel SVM')
        gauss_ax.set_xlabel('Order of Polynomial')
        gauss_ax.set_ylabel('Success Rate')
        fig.savefig('poly_order.png')
        
    def poly_svm(self, order):
        kernel = PyML.ker.Polynomial(order)
        svm = PyML.classifiers.multi.OneAgainstOne(PyML.SVM(kernel))

        svm.train(self.trainingpymldata)

        svm_results = svm.test(self.testingpymldata)
        
        return svm_results

if __name__ == "__main__":
    rt = ReportTests()
    rt.plot_gauss_gamma()
    rt.plot_gauss_gamma_log()
    #rt.opt_gauss_gamma()
    rt.do_linear_svm_test()
    rt.do_polynomial_svm_test()


#resf.close()

# TEST ONE AGAINST MANY AND ONE AGAINST ONE
# TEST KNN AND RIGDE REGRESSION
# TEST TIME OF ALL CLASSIFIERS
# TEST CLASSIFIERS WITH DIFFERING PARAMETERS
#   -   EG SVM WITH POLYNOMIAL KERNEL WITH DIFFERENT ORDER POLYNOMIAL
#   -   MINIMISE OVER THIS