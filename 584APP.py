# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 16:28:06 2015

@author: LT
"""

# Temperature-conversion program using PyQt
import pandas as pd
import sys
from PyQt4 import QtGui,uic,QtCore
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
        self.statistic={"win":0,"winvalue":0,"lose":0,"losevalue":0,"P/L":0,"Profit factor":0}
        self.windowT = int(self.window.text())
        #connect button with event
        self.pushButton.clicked.connect(self.plot)
        self.pushButton_2.clicked.connect(self.plot_2)
        self.pushButton_3.clicked.connect(self.plot_3)
        self.pushButton_4.clicked.connect(self.plot_4)
        #connect to tableview
        self.tableData = QtGui.QStandardItemModel(1, 6)
        #tableData.setData(tableData.index(0, 0), QtCore.QVariant("SB"))
        for  n, key in enumerate(sorted(self.statistic.keys())):
            self.tableData.setHeaderData(n, QtCore.Qt.Horizontal, key)
            self.tableData.setData(self.tableData.index(0, n), str(self.statistic[key]))
        """
        tableData.setHeaderData(0, QtCore.Qt.Horizontal, "N of win")
        tableData.setHeaderData(1, QtCore.Qt.Horizontal, "winvalue")
        tableData.setHeaderData(2, QtCore.Qt.Horizontal, "N of lose")
        tableData.setHeaderData(3, QtCore.Qt.Horizontal, "losevalue")
        tableData.setHeaderData(4, QtCore.Qt.Horizontal, "P/L")
        tableData.setHeaderData(5, QtCore.Qt.Horizontal, "Profit factor")
        """
        self.tableView_static.setModel(self.tableData)
        
        

  
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
        mypair = Pairstrading.SinglePairstradingStrategy(pair,bars,buy_signal=self.buy_signal.text()
                                                        ,sell_signal=self.sell_signal.text(), window=self.window.text())
        signals = mypair.generate_signals()
        # Create a portfolio of pairtrading, with $100,000 initial capital
        portfolio = Pairstrading.PairstradingPortfolio(pair, bars, signals, initial_capital=100000.0)
        returns = portfolio.backtest_portfolio()
        self.statistic = portfolio.backtest_statistic()
        AtoB = mypair.generate_AtoB()
        
        ax = self.figure_3.add_subplot(211)
        
        
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
        """
        ax = self.figure_3.add_subplot(211)       
        ax.hold(False) 
        ax.plot(AtoB.index,AtoB['A/B'])
        ax.hold(True)
        
        plt.ylabel(pair[0]+'/'+ pair[1])
        #add moving average
        mavg = pd.rolling_mean(AtoB['A/B'], self.windowT, min_periods=1)
        mavg.plot(ax=ax, lw=2.)
        """
        
                 
        ax2 = self.figure_3.add_subplot(212)
        # discards the old graph
        ax2.hold(False)
        # plot data
                
        ax2.plot(returns.index,returns['total'],label='Portfolio',color='r')                  
        ax2.hold(True)
        ax2.plot(returns.index,returns['SPY'],label='SPY') 
        plt.legend(loc=0)
        plt.ylabel('Portfolio value in $')
        # Plot the "buy" and "sell" trades against the equity curve
        """
        ax2.plot(returns.ix[signals[pair[0]] > 0].index, 
                 returns.total[signals[pair[0]] > 0],
                 '^', markersize=10, color='m')
        ax2.plot(returns.ix[signals[pair[0]] < 0].index, 
                 returns.total[signals[pair[0]] < 0],
                 'v', markersize=10, color='k')
        """
        # refresh canvas          
        self.canvas_3.draw()
        ##refresh statistc data
        for  n, key in enumerate(sorted(self.statistic.keys())):
            self.tableData.setData(self.tableData.index(0, n), str(self.statistic[key]))
        

    
    def plot_4(self):           #backtest result
        pair = (self.lineEdit.text(),self.lineEdit_2.text())
        bars = Pairstrading.generate_bars(pair)
        #generate trading signals
        mypair = Pairstrading.SinglePairstradingStrategy(pair,bars,buy_signal=self.buy_signal.text()
                                                        ,sell_signal=self.sell_signal.text(), window=self.window.text())
        signals = -mypair.generate_signals()
        # Create a portfolio of pairtrading, with $100,000 initial capital
        portfolio = Pairstrading.PairstradingPortfolio(pair, bars, signals, initial_capital=100000.0)
        returns = portfolio.backtest_portfolio()
        self.statistic = portfolio.backtest_statistic()
        AtoB = mypair.generate_AtoB()
        
        ax = self.figure_3.add_subplot(211)
        
        
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

        
                 
        ax2 = self.figure_3.add_subplot(212)
        # discards the old graph
        ax2.hold(False)
        # plot data
               
        ax2.plot(returns.index,returns['total'],label='Portfolio',color='r')                  
        ax2.hold(True)
        ax2.plot(returns.index,returns['SPY'],label='SPY') 
        plt.legend(loc=0)
        
        plt.ylabel('Portfolio value in $')
        """
        # Plot the "buy" and "sell" trades against the equity curve
        ax2.plot(returns.ix[signals[pair[0]] > 0].index, 
                 returns.total[signals[pair[0]] > 0],
                 '^', markersize=10, color='m')
        ax2.plot(returns.ix[signals[pair[0]] < 0].index, 
                 returns.total[signals[pair[0]] < 0],
                 'v', markersize=10, color='k')
        # refresh canvas 
        """
        self.canvas_3.draw()
        ##refresh statistc data
        for  n, key in enumerate(sorted(self.statistic.keys())):
            self.tableData.setData(self.tableData.index(0, n), str(self.statistic[key]))
        
class StatistiTable(QtGui.QStandardItemModel):
    def __init__(self, data, *args):
        QtGui.QTableWidget.__init__(self, *args)
        self.data = data
        self.setmydata()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
 
    def setmydata(self): 
        horHeaders = []
        for n, key in enumerate(sorted(self.data.keys())):
            horHeaders.append(key)
            item = str(self.data[key])
            newitem = QtGui.QTableWidgetItem(item)
            self.setItem(0, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)        
     

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myWindow = MyWindowClass(None)
    myWindow.show()
    app.exec_()