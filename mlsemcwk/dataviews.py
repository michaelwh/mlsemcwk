#!/usr/bin/env python
# Tk-matplotlib integration code from http://matplotlib.sourceforge.net/examples/user_interfaces/embedding_in_tk2.html
import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot

import Tkinter as Tk
import sys

import PyML as pyml

import semdatautil

def destroy(e): sys.exit()

class MatplotlibTkFigFrame(object):
    
    def __init__(self, fig):
        self.fig = fig
        ## ------------------------- TK STUFF
        
        self.root = Tk.Tk()
        self.root.wm_title("MatplotlibTkFigFrame")
        #root.bind("<Destroy>", destroy)
        
        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        #toolbar = NavigationToolbar2TkAgg( canvas, root )
        #toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^ TK STUFF
    
    def run_tk_mainloop(self):
        Tk.mainloop()

class DataViewer(MatplotlibTkFigFrame):
    def __init__(self, fig, ax, datarows, actuallabels=None, predictedlabels=None, startno=0):
        super(DataViewer, self).__init__(fig)
        self.currdatano = 0
        self.fig = fig
        self.ax = ax
        self.datarows = datarows
        self.actuallabels = actuallabels
        self.predictedlabels = predictedlabels
        
        self.quitbutton = Tk.Button(master=self.root, text='Quit', command=sys.exit)
        self.quitbutton.pack(side=Tk.BOTTOM)
        
        self.nextbutton = Tk.Button(master=self.root, text='>', command=self.next_data)
        self.nextbutton.pack(side=Tk.BOTTOM)
        
        self.prevbutton = Tk.Button(master=self.root, text='<', command=self.prev_data)
        self.prevbutton.pack(side=Tk.BOTTOM)
  
        self.show_data(startno)
        
        super(DataViewer, self).run_tk_mainloop()
    
    def show_data(self, datano):
        self.currdatano = datano
        self.ax.imshow(semdatautil.sem_datarow_to_image(self.datarows[datano]), cmap=pyplot.gray())
        self.fig.canvas.draw()
        print self.currdatano
        if self.actuallabels != None:
            print "Actual Label: " + str(self.actuallabels[self.currdatano])
            
        if self.predictedlabels != None:
            print "Predicted Label: " + str(self.predictedlabels[self.currdatano])
        
    def next_data(self):
        self.show_data(self.currdatano + 1)
    
    def prev_data(self):
        if self.currdatano > 0:
            self.show_data(self.currdatano - 1)