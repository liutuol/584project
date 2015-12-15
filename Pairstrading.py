# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 18:43:31 2015

@author: LT
"""

# Pairstrading.py
import pandas as pd
import numpy as np
from backtest import Portfolio,Strategy
from getyahooandstore import get_price
import matplotlib.pyplot as plt




def generate_bars(pair,START = "2001-01-01"):
    """this function is used to get bars(adjclose) from pair"""    

    bars = pd.DataFrame([])
    bars[pair[0]] = get_price(pair[0], START)
    bars[pair[1]] = get_price(pair[1], START)
    return bars
    
class SinglePairstradingStrategy(Strategy):
    """    
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    buy_signal - bigger than it to buy
    sell_signal - less than it to sell
    window - Lookback period for MA and std.
    long_window - Lookback period for long moving average.

    """

    def __init__(self, pair, bars, buy_signal=-2,sell_signal=2, window=30):
        self.pair = pair
        self.bars = bars
        self.buy_signal = float(buy_signal)
        self.sell_signal = float(sell_signal)
        self.window = int(window)



    def generate_signals(self): 
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        
        signals['Longshortstatues'] = 0.0
        signals['tradesignal'] = 0.0
        signals[self.pair[0]] = 0.0
        signals[self.pair[1]] = 0.0
        #create signal 
        AtoB = self.generate_AtoB()
         

        # Create a 'signal' buy when AtoB['delta']>buy_signal, sell when AtoB['delta']<sell_signal
        if self.sell_signal > self.buy_signal:
            signals['signal'][self.window:] = AtoB['delta'][self.window:].apply(
            lambda x: -1 if x > self.sell_signal else (1 if x < self.buy_signal else 0))
        else:
            signals['signal'][self.window:] = AtoB['delta'][self.window:].apply(
            lambda x: -1 if x < self.sell_signal else (1 if x > self.buy_signal else 0))
        
        
        for i in range(self.window,len(signals)):                         
            signals['Longshortstatues'][i] = signals['Longshortstatues'][i-1] + signals['tradesignal'][i-1]
            if signals['signal'][i] != 0 :
                signals['tradesignal'][i] = signals['signal'][i] - signals['Longshortstatues'][i]
        
        
        # generate signal for stock,this is trading signal not position
        
        signals[self.pair[0]] = signals['tradesignal']
        
        signals[self.pair[1]] = -signals['tradesignal'] * AtoB['MA']
        
        
        
        

        return signals.loc[:,self.pair]
    def generate_signals_MA(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['tradesignal'] = 0.0
        signals['Longshortstatues'] = 0.0
        signals[self.pair[0]] = 0.0
        signals[self.pair[1]] = 0.0
        #create signal 
        AtoB = self.generate_AtoB()
        short_window = 10
        long_window = 30 
        signals['short_mavg'] = pd.rolling_mean( AtoB['A/B'], short_window, min_periods=1)
        signals['long_mavg'] = pd.rolling_mean( AtoB['A/B'], long_window, min_periods=1)       
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
            > signals['long_mavg'][short_window:], 1.0, 0.0)   

        # Take the difference of the signals in order to generate actual trading orders
        signals['tradesignal'] = signals['signal'].diff()         
        
        # generate signal for stock,this is trading signal not position
        
        signals[self.pair[0]] = signals['tradesignal']
        
        signals[self.pair[1]] = -signals['tradesignal'] * AtoB['MA']
        #the last one is not good, deal it when generat position set last one to zero
        return signals.loc[:,self.pair]
    def generate_AtoB(self):
                #create signal 
        AtoB = pd.DataFrame(index=self.bars.index)
        AtoB['A/B'] = self.bars[self.pair[0]] / self.bars[self.pair[1]]
        AtoB['MA'] = pd.rolling_mean(AtoB['A/B'], self.window, min_periods=1)
        AtoB['std'] = pd.rolling_std(AtoB['A/B'], self.window)
        AtoB['delta'] = (AtoB['A/B']-AtoB['MA'])/AtoB['std'] 
        return AtoB

        
      
 

      
################################################################################################################################
        ##########above is strategy$#3###########################################
        ###################################################################
        


class PairstradingPortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, pair, bars, signals, initial_capital=1000000.0):
        self.pair = pair        
        self.bars = bars
        self.signals = signals            #the signals is buy sell signal not position
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.bars.index,columns=self.bars.columns).fillna(0.0) 
        cumsignal = self.signals.cumsum()
        paircost = (abs(self.signals)*self.bars).sum(axis=1)/2   #cost per pair  
        for j in range(1,len(positions.index)):
            if paircost.iloc[j] !=0:                
                N_pair = self.initial_capital / paircost.iloc[j]
                positions.iloc[j] = N_pair * cumsignal.iloc[j]
            else:                    
                positions.iloc[j] = positions.iloc[j-1]
                           
          # This strategy buys 1 shares
        positions.iloc[-1,:]=0        
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.signals.index) 

        pos_diff = self.positions.diff().fillna(0.0)               
        #CaculationMatrix['pair']
        
        portfolio['holdings'] = (self.positions*self.bars).sum(axis=1)
        portfolio['cash'] = self.initial_capital - ((pos_diff*self.bars).sum(axis=1)).cumsum()

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        portfolio['SPY'] = get_price('SPY',self.signals.index[0])
        portfolio['SPY'] = self.initial_capital/portfolio['SPY'][0]*portfolio['SPY']
        
        return portfolio
        
    def backtest_statistic(self):
        portfolio = self.backtest_portfolio()
        statistic = {}
        """tradeornot = pd.DataFrame(index=signals.index)
        tradeornot["trade"] = 0
        tradeornot[self.signals[self.symbols[0]]<>0] = 1"""
        
        earnpertrade = portfolio[self.signals[self.pair[0]]<>0].diff().fillna(0.0)['total']
        statistic["win"] = earnpertrade[earnpertrade > 0].count()
        statistic["winvalue"] = round(earnpertrade[earnpertrade > 0].sum(),2)
        statistic["lose"] = earnpertrade[earnpertrade < 0].count()
        statistic["losevalue"] = round(earnpertrade[earnpertrade < 0].sum(),2)
        statistic["P/L"] = statistic["winvalue"] + statistic["losevalue"]
        statistic["Profit factor"] = round(-statistic["winvalue"] / statistic["losevalue"],2)
        return statistic
        
   

class Problem1Portfolio(PairstradingPortfolio):
    """buy or sell 1 for each pair.
    Requires:
    pairs- list of tuple  
    bars  - dataframe timeserierse
    signals - list of dataframe
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, bars, signals, initial_capital=1000000.0):
                #list of tuple        
        self.bars = bars          #dataframe
        self.signals = signals     #list of data frame
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.bars.index,columns=self.bars.columns).fillna(0.0) 
        
        for signal in self.signals:
            positions[signal.columns] = positions[signal.columns] + 1*signal
        
        
        positions = positions.cumsum()
        positions.iloc[-1,:] = 0
        return positions
                    
    def backtest_portfolio(self):
               
        portfolio = pd.DataFrame(index=self.bars.index) 

        pos_diff = self.positions.diff()        
        #CaculationMatrix['pair']
        
        portfolio['holdings'] = (self.positions*self.bars).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)     
        
        
        

        
        portfolio['SPY'] = get_price('SPY',self.bars.index[0])
        portfolio['SPY'] = self.initial_capital/portfolio['SPY'][0]*portfolio['SPY']
        
        return portfolio
    
    
    def backtest_statistic(self):
        portfolio = self.backtest_portfolio()
        statistic = {}
        """tradeornot = pd.DataFrame(index=signals.index)
        tradeornot["trade"] = 0
        tradeornot[self.signals[self.symbols[0]]<>0] = 1"""
        
        earnpertrade = portfolio[self.positions.diff().sum(axis=1)<>0].diff().fillna(0.0)['total']
        statistic["win"] = earnpertrade[earnpertrade > 0].count()
        statistic["winvalue"] = round(earnpertrade[earnpertrade > 0].sum(),2)
        statistic["lose"] = earnpertrade[earnpertrade < 0].count()
        statistic["losevalue"] = round(earnpertrade[earnpertrade < 0].sum(),2)
        statistic["P/L"] = statistic["winvalue"] + statistic["losevalue"]
        if statistic["lose"] == 0:
            statistic["Profit factor"] = 1
        else:
            statistic["Profit factor"] = round(-statistic["winvalue"] / statistic["losevalue"],2)      
        return statistic

class Problem3Portfolio(Problem1Portfolio):
    
    def generate_positions(self):
        positions = pd.DataFrame(index=self.bars.index,columns=self.bars.columns).fillna(0.0)           
        
        percent = np.array((0.25,0.25,0.25,0.25))       
        pairmoney = percent * self.initial_capital
        
        for i in range(len(self.signals)):
            cumsignal = self.signals[i].cumsum()
            paircost = (abs(self.signals[i])*self.bars[self.signals[i].columns]).sum(axis=1)/2   #cost per pair  
            for j in range(1,len(positions.index)):
                if paircost.iloc[j] !=0:                
                    N_pair = pairmoney[i] / paircost.iloc[j] *0.3
                    positions.iloc[j][self.signals[i].columns] = N_pair * cumsignal.iloc[j]
                else:                    
                    positions.iloc[j][self.signals[i].columns] = positions.iloc[j-1][self.signals[i].columns]
            """
            multiplefactor.loc[:,self.signals[i].columns[0]] = np.where(self.signals[i].sum(axis=1)!=0, pairmoney[i] / paircost,0)
            multiplefactor.loc[:,self.signals[i].columns[1]] = multiplefactor.loc[:,self.signals[i].columns[0]]
        
        lastmult = pd.Series(multiplefactor.iloc[0,:])
        for mult in multiplefactor.itertuples(): 
            index = np.array(mult[1:])
            index = (index!=0)
            multiplefactor.loc[mult[0],index] = (multiplefactor.loc[mult[0],index] + lastmult[index])/2       
                         
            lastmult[index] = multiplefactor.loc[mult[0],index]
        
        positions = positions * multiplefactor
        self.multiplefactor = multiplefactor
        positions = positions.cumsum()
        """
        positions.iloc[-1,:] = 0
        return positions
class Problem3Portfolio_rebalance(Problem1Portfolio):
    def __init__(self, bars, signals, initial_capital=1000000.0,lookwindow = 20):
        self.lookwindow = int(lookwindow)
        self.bars = bars          #dataframe
        self.signals = signals     #list of data frame
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
    def generate_positions(self):
        # performance 
        
        performance = []
        for signal in self.signals:
            holding = ((signal.cumsum())*self.bars).sum(axis=1)
            cash = 2000 - (signal*self.bars).sum(axis=1).cumsum()
            total = holding +cash
            total = total.pct_change().fillna(0.0)
            total = pd.rolling_mean( total, self.lookwindow, min_periods=1).fillna(0.0)
            performance.append(total)
        performance = pd.DataFrame(performance)
        performance = performance.transpose()
        self.performance = pd.DataFrame(performance)
        
            
            
        positions = pd.DataFrame(index=self.bars.index,columns=self.bars.columns).fillna(0.0)           
        
        percent = np.array((0.25,0.25,0.25,0.25))       
        pairmoney = percent * self.initial_capital
        
        for i in range(len(self.signals)):
            cumsignal = self.signals[i].cumsum()
            paircost = (abs(self.signals[i])*self.bars[self.signals[i].columns]).sum(axis=1)/2   #cost per pair  
            for j in range(1,len(positions.index)):
                if j % self.lookwindow  == 0:
                    perform = performance.iloc[j,]
                    if perform.rank()[i] > 2:
                        pairmoney[i] = 0.4 * self.initial_capital
                    else:
                        pairmoney[i] = 0.1 * self.initial_capital
                if paircost.iloc[j] !=0:                
                    N_pair = pairmoney[i] / paircost.iloc[j]*0.3
                    positions.iloc[j][self.signals[i].columns] = N_pair * cumsignal.iloc[j]
                else:                    
                    positions.iloc[j][self.signals[i].columns] = positions.iloc[j-1][self.signals[i].columns]


    
        positions.iloc[-1,:] = 0
        return positions

def multiplepair_signalgenerate(mypairs,buysignals,sellsignals,window=30):
    multibars = pd.DataFrame()
    multisignals = []            # list of dataframe
    for i in range(len(mypairs)):
        bars = generate_bars(mypairs[i])               
        singlepair = SinglePairstradingStrategy(mypairs[i],bars,buy_signal=buysignals[i],sell_signal=sellsignals[i], window=window)
        multisignals.append(singlepair.generate_signals())
        multibars = multibars.combine_first(bars)
        #multisignals[i] = multisignals.append(signals)
    return multibars,multisignals
    
def multiplepair_signalgenerate_problem2(mypairs,buysignals,sellsignals,window=30):
    #tuple ,every strock use the same signal
    multibars = pd.DataFrame()
    multisignals = []            # list of dataframe
    for i in range(len(mypairs)):
        bars = generate_bars(mypairs[i])
        multibars = multibars.combine_first(bars)
        status = pd.DataFrame(index=bars.index,columns=bars.columns).fillna(0.0)             
        for j in range(len(buysignals)):       
            singlepair = SinglePairstradingStrategy(mypairs[i],bars,buy_signal=buysignals[j],sell_signal=sellsignals[j], window=window)
            newstatus = (singlepair.generate_signals()).cumsum()             
            status = status + newstatus        
        
        multisignals.append(status.diff())
        
        #multisignals[i] = multisignals.append(signals)
    return multibars,multisignals    
    
def multiplepair_signalgenerate_problem2_MA(mypairs,buysignals,sellsignals,window=30):
    #tuple ,every strock use the same signal
    multibars = pd.DataFrame()
    multisignals = []            # list of dataframe
    for i in range(len(mypairs)):
        bars = generate_bars(mypairs[i])
        multibars = multibars.combine_first(bars)
        status = pd.DataFrame(index=bars.index,columns=bars.columns).fillna(0.0)             
        for j in range(len(buysignals)):       
            singlepair = SinglePairstradingStrategy(mypairs[i],bars,buy_signal=buysignals[j],sell_signal=sellsignals[j], window=window)
            newstatus = (singlepair.generate_signals()).cumsum()             
            status = status + newstatus        
        status_ma = (singlepair.generate_signals_MA()).cumsum()
        status = status * status_ma
        multisignals.append(status.diff())
        
        #multisignals[i] = multisignals.append(signals)
    return multibars,multisignals        
##############################################################
        ########## test       ######################
################################################
        
        
        
def test_signalpair():
    pair = ("SPY","AAPL")
    bars = generate_bars(pair)
    #generate trading signals
    #mypair = SinglePairstradingStrategy(pair,bars,buy_signal=1,sell_signal=-2, window=30)
    mypair = SinglePairstradingStrategy(pair,bars,buy_signal=-2,sell_signal=2, window=30)
    signals = mypair.generate_signals() + (mypair.generate_signals_MA().cumsum()) #
    
    # Create a portfolio of pairtrading, with $100,000 initial capital
    portfolio = PairstradingPortfolio(pair, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    statistic = portfolio.backtest_statistic()
    AtoB = mypair.generate_AtoB()
    
    # Plot two charts to assess trades and equity curve
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    
     # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    # Plot the "buy" and "sell" trades against the equity curve
    ax2.plot(returns.ix[signals[pair[0]] > 0].index, 
             returns.total[signals[pair[0]] > 0],
             '^', markersize=10, color='m')
    ax2.plot(returns.ix[signals[pair[0]] < 0].index, 
             returns.total[signals[pair[0]] < 0],
             'v', markersize=10, color='k')

    # Plot the figure
    fig.show()
    
    
def test_problem1():
    mypairs = [('AAPL','AMZN'),('INTC','MA'),('FISV','GOOGL'),('ADP','AVGO')]
    buysignals = (-2,-2,-2,-2)
    sellsignals = (2,2,2,2)
    multibars,multisignals = multiplepair_signalgenerate(mypairs,buysignals,sellsignals)
    
    # Create a portfolio of pairtrading, with $100,000 initial capital
    portfolio = Problem1Portfolio(multibars, multisignals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    statistic = portfolio.backtest_statistic()
    
    
    # Plot two charts to assess trades and equity curve
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    
     # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)



    # Plot the figure
    fig.show()
    
def test_problem2():
    mypairs = [('AAPL','AMZN'),('INTC','MA'),('FISV','GOOGL'),('ADP','AVGO')]
    buysignals = (2,2.2,2.4)
    sellsignals = (-2,-2.2,-2.4)
    multibars,multisignals = multiplepair_signalgenerate_problem2(mypairs,buysignals,sellsignals)
    
    # Create a portfolio of pairtrading, with $100,000 initial capital
    portfolio = Problem1Portfolio(multibars, multisignals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    statistic = portfolio.backtest_statistic()
    
    
    # Plot two charts to assess trades and equity curve
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    
     # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)



    # Plot the figure
    fig.show()
    
def test_multiplepair_signalgenerate():
    mypairs = [('AAPL','AMZN'),('INTC','MA'),('FISV','GOOGL'),('ADP','AVGO')]
    buysignals = (-2,-2,-2,-2)
    sellsignals = (2,2,2,2)
    multibars,multisignals = multiplepair_signalgenerate(mypairs,buysignals,sellsignals)
    
    
    
       
if __name__ == '__main__':
    mypairs = [('AAPL','AMZN'),('INTC','MA'),('FISV','GOOGL'),('ADP','AVGO')]
    buysignals = (2,2.2,2.4)
    sellsignals = (-2,-2.2,-2.4)
    multibars,multisignals = multiplepair_signalgenerate_problem2_MA(mypairs,buysignals,sellsignals)
    
    # Create a portfolio of pairtrading, with $100,000 initial capital
    portfolio = Problem3Portfolio_rebalance(multibars, multisignals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    statistic = portfolio.backtest_statistic()
    
    
    # Plot two charts to assess trades and equity curve
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    
     # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)



    # Plot the figure
    fig.show()
    