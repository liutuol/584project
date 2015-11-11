# 584project
At now, there are 4 py files, and 2 data files,1 UI file.
##(1) getyahooandstore.py         
 in this file, it can scrape the sap500 ticker list and store it in file sector_tickers.csv. Then download the daily OHLC price from yahoo finance for sap500 stocks, and store them in file SNP_500.h5. This file still offer a function get_price(),which get price from local file, or if local file doesn’t exist, get if from yahoo finance.
##(2) backtest.py              
 it’s the basic class, contain 2 virtual class Strategy and Portfolio. Strategy is an abstract base class for trading strategies, Portfolio is an abstract base class for backtesting.
##(3) pairstrading.py           
 Derive pairs trading strategy and back test class from abstract base classes.
##(4) 584APP.py       
 combine all things together. It connects to PriceUI.ui(this UI file is created by PYQT designer), display a simple interface, that user can change stocks and use the predefined pairs trading strategy to back test it.
###The project contain 3 questions, now is in 1st question.
