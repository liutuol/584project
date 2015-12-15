# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 15:28:02 2015

@author: LT
"""

import urllib2
import pytz
import pandas as pd
import os.path

from bs4 import BeautifulSoup
from datetime import datetime
from pandas.io.data import DataReader
SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"




def scrape_list(site):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(site, headers=hdr)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page)
    if os.path.exists('sector_tickers.csv'):
        print ("Read tikckers from local mechine")
        sector_tickers = pd.read_csv('sector_tickers.csv')
        
    else:
        print ("No file exist, get from web")
        table = soup.find('table', {'class': 'wikitable sortable'})
        sector_tickers = []
        for row in table.findAll('tr'):
            col = row.findAll('td')
            if len(col) > 0:
                sector = str(col[3].string.strip()).lower().replace(' ', '_')
                ticker = str(col[0].string.strip())
                sector_tickers.append((sector,ticker))            
                """if sector not in sector_tickers:
                    sector_tickers[sector] = list()
                sector_tickers[sector].append(ticker)"""
        sector_tickers.append(('Benchmark','SPY'))
        sector_tickers = pd.DataFrame(sector_tickers, columns=['sector', 'ticker'])        
        sector_tickers.to_csv('sector_tickers.csv')
    return sector_tickers


def download_ohlc(sector_tickers, start, end):
    sector_ohlc = {}
    for tickers in sector_tickers["ticker"]:
        print 'Downloading data from Yahoo for %s ' % tickers
        try:
            data = DataReader(tickers, 'yahoo', start, end)
            sector_ohlc[tickers] = data
        except IOError:
            print 'Someproblem wiht downloading for %s ' % tickers
        
    print 'Finished downloading data'
    return sector_ohlc


def store_HDF5(sector_ohlc, path):
    with pd.get_store(path) as store:
        for tickers, ohlc in sector_ohlc.iteritems():
            store[tickers] = ohlc


def get_snp500():
    
    START = datetime(2000, 1, 1, 0, 0, 0, 0, pytz.utc)
    END = datetime.today().utcnow()
    sector_tickers = scrape_list(SITE)
    sector_ohlc = download_ohlc(sector_tickers, START, END)
    store_HDF5(sector_ohlc, 'SNP_500.h5')

def get_price(ticker, start, end=datetime.today().utcnow(),pricetype="Adj Close"):
    if os.path.exists('SNP_500.h5'):
        #print 'read %s from local mechine' % ticker 
        store = pd.HDFStore('SNP_500.h5')        
        price = store[ticker][pricetype]
        store.close          
    else:
        #print 'connect to network,read %s from yahoo' % ticker
        data = DataReader(ticker, 'yahoo', start, end)
        price = data[pricetype]
    return price.loc[start:end]
    


if __name__ == '__main__':
    AAPL=get_price("AAPL", "2003-01-21")
    print AAPL
    #sector_tickers = scrape_list(SITE)
    #get_snp500()
    
    #import h5py
    #f=h5py.File('snp500.h5', 'r')
    #dset=f['energy']
    #store = pd.HDFStore('snp500.h5')
    """
    pric
    start = datetime(2005, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime.today().utcnow()
    print(store)
    w=store['industrials']
    z=w.ix[:,:,'AAL']
    z=w.ix[:,-1,-1]
    w.ix[:,1,2].name      #the name is the stock name
    z=w.ix["close",2,2]   #get "close" price for stock 2 at time 2
    z=w.ix["close",:,2]   #get "close" price for stock 2 at all time in database
    z.index
    z.index[-1]    #get the last date
    z.name          #get stock name """
    
    