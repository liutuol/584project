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



def generate_bars(pair,START = "2005-01-01"):
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

    def __init__(self, pair, bars, buy_signal=2,sell_signal=-2, window=30):
        self.pair = pair
        self.bars = bars
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.window = window
      

    def generate_signals(self):
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
         

        # Create a 'signal' buy when AtoB['delta']>buy_signal, sell when AtoB['delta']<sell_signal
        
        signals['signal'][self.window:] = AtoB['delta'][self.window:].apply(
        lambda x: 1 if x > self.buy_signal else (-1 if x < self.sell_signal else 0))  

        # Take the difference of the signals in order to generate actual trading orders for pair
        for i in range(self.window,len(signals)):                         
            signals['Longshortstatues'][i] = signals['Longshortstatues'][i-1] + signals['tradesignal'][i-1]
            if signals['signal'][i] != 0 :
                signals['tradesignal'][i] = signals['signal'][i] - signals['Longshortstatues'][i]
        # generate signal for stock,this is trading signal not position
        signals[self.pair[0]] = signals['tradesignal']
        signals[self.pair[1]] = -signals['tradesignal'] * AtoB['MA']
        

        return signals.loc[:,self.pair]
    def generate_AtoB(self):
                #create signal 
        AtoB = pd.DataFrame(index=self.bars.index)
        AtoB['A/B'] = self.bars[self.pair[0]] / self.bars[self.pair[1]]
        AtoB['MA'] = pd.rolling_mean(AtoB['A/B'], self.window, min_periods=1)
        AtoB['std'] = pd.rolling_std(AtoB['A/B'], self.window)
        AtoB['delta'] = (AtoB['A/B']-AtoB['MA'])/AtoB['std'] 
        return AtoB
        

        


class PairstradingPortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbols, bars, signals, initial_capital=1000000.0):
        self.symbols = symbols        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        for symbol in self.symbols:            
            positions[symbol] = 100*self.signals[symbol].cumsum()   # This strategy buys 1 shares
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.signals.index) 
        """tradeornot = pd.DataFrame(index=signals.index)
        tradeornot = 0
        tradeornot = self.signals[symbols[0]].apply(lambda x: 1 if x <> 0 else 0)  
        """
        for symbol in self.symbols:
            portfolio[symbol] = self.positions[symbol]*self.bars[symbol]
        pos_diff = self.positions.diff().fillna(0.0)               
        

        portfolio['holdings'] = (self.positions*self.bars).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars).sum(axis=1).cumsum()

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

       
if __name__ == '__main__':
    pair = ("AAPL","GOOGL")
    bars = generate_bars(pair)
    #generate trading signals
    mypair = SinglePairstradingStrategy(pair,bars)
    signals = mypair.generate_signals()
    
    # Create a portfolio of pairtrading, with $100,000 initial capital
    portfolio = PairstradingPortfolio(pair, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    
    
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

    
    