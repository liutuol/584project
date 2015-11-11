# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 16:28:06 2015

@author: LT
"""

# Temperature-conversion program using PyQt
 
import sys
from PyQt4 import QtGui,uic
from getyahooandstore import get_price
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt
import Pairstrading

START = "2001-01-01"
form_class = uic.loadUiType("PriceUI.ui")[0]                 # Load the UI
 
class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        """self.btn_CtoF.clicked.connect(self.btn_CtoF_clicked)  # Bind the event handlers
        self.btn_FtoC.clicked.connect(self.btn_FtoC_clicked)  #   to the buttons
 """
 
        #connect button with event
        self.pushButton.clicked.connect(self.plot)
        self.pushButton_2.clicked.connect(self.plot_2)
        self.pushButton_3.clicked.connect(self.plot_3)
        #connect to figure
        
        self.figure = plt.figure()
        self.figure_2 = plt.figure()
        self.figure_3 = plt.figure()
        #plot1

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)       
        self.plotlayout.addWidget(self.toolbar)
        self.plotlayout.addWidget(self.canvas)

        #plot2
        #self.figure_2 = plt.figure()
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.toolbar_2 = NavigationToolbar(self.canvas_2, self)   
        self.plotlayout_2.addWidget(self.toolbar_2)
        self.plotlayout_2.addWidget(self.canvas_2)
        
        #plot2
        #self.figure_2 = plt.figure()
        self.canvas_3 = FigureCanvas(self.figure_3)
        self.toolbar_3 = NavigationToolbar(self.canvas_3, self)   
        self.plotlayout_3.addWidget(self.toolbar_3)
        self.plotlayout_3.addWidget(self.canvas_3)        

        
    def plot(self):
        ticker = self.lineEdit.text()        
        data = get_price(ticker, START)
        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        ax.hold(False)
        # plot data
        ax.plot(data.index,data)
        # refresh canvas
        self.canvas.draw()
    def plot_2(self):
        ticker = self.lineEdit_2.text()        
        data = get_price(ticker, START)
        # create an axis
        ax = self.figure_2.add_subplot(111)
        # discards the old graph
        ax.hold(False)
        # plot data
        ax.plot(data.index,data)
        # refresh canvas
        self.canvas_2.draw()   
    
    def plot_3(self):           #backtest result
        pair = (self.lineEdit.text(),self.lineEdit_2.text())
        bars = Pairstrading.generate_bars(pair)
        #generate trading signals
        mypair = Pairstrading.SinglePairstradingStrategy(pair,bars)
        signals = mypair.generate_signals()
        # Create a portfolio of pairtrading, with $100,000 initial capital
        portfolio = Pairstrading.PairstradingPortfolio(pair, bars, signals, initial_capital=100000.0)
        returns = portfolio.backtest_portfolio()
        ax = self.figure_3.add_subplot(211)
        
        AtoB = mypair.generate_AtoB()
        # discards the old graph
        ax.hold(False)
        # plot data
        ax.plot(AtoB.index,AtoB['delta'])
        ax.hold(True)
        plt.ylabel('stock1/stock2 Z')
        # Plot the "buy" and "sell" trades against the equity curve
        ax.plot(AtoB.ix[signals[pair[0]] > 0].index, 
                 AtoB.delta[signals[pair[0]] > 0],
                 '^', markersize=10, color='m')
        ax.plot(AtoB.ix[signals[pair[0]] < 0].index, 
                 AtoB.delta[signals[pair[0]] < 0],
                 'v', markersize=10, color='k')
        # create an axis
        ax2 = self.figure_3.add_subplot(212)
        # discards the old graph
        ax2.hold(False)
        # plot data
        ax2.plot(returns.index,returns['total'])
        ax2.hold(True)
        plt.ylabel('Portfolio value in $')
        # Plot the "buy" and "sell" trades against the equity curve
        ax2.plot(returns.ix[signals[pair[0]] > 0].index, 
                 returns.total[signals[pair[0]] > 0],
                 '^', markersize=10, color='m')
        ax2.plot(returns.ix[signals[pair[0]] < 0].index, 
                 returns.total[signals[pair[0]] < 0],
                 'v', markersize=10, color='k')
        # refresh canvas
        self.canvas_3.draw()          
     

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myWindow = MyWindowClass(None)
    myWindow.show()
    app.exec_()