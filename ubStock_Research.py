import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from numba import jit
from pymongo import MongoClient
from yahoo_fin import stock_info as si
from typing import Union



def main():
    """
    ubStock_Research, yet another stock research module
    Utility_Belt Designs, Tacoma, WA
    @author: ZennDogg, with a lot of help for find_stock_correlation from outside sources
    python 3.10.


    class StockSearchSupport: Contains support functions for class CorrelatedStockResearch.

    Function List:

        -  get_info                     -  Retrieves stock information from yfinance. Used by several functions
        -  get_dividend                 -  Retrieves dividend information from yfinance. Used by several functions
        -  find_interval                -  Determines the interval between dividend payments (in months)
        -  interval_range               -  Transforms the return of find_interval into a single digit integer
        -  gather_all_stock_symbols     -  Finds stock symbols from the Dow, SP500, NASDAQ and others.


    class CorrelatedStockResearch:

    Ultimately, prunes 8500+ stock symbols down to 10 pairs of stocks with a high degree of non-correlation.
    Corrolated stocks move together (same or opposite direction). Uncorrolated stocks move independently of
    each other. Those stocks are then run through a recommendation algorithm and sorted by highest recommendation score.
    see:  https://www.investopedia.com/articles/financial-theory/09/uncorrelated-assets-diversification.asp

    All data storage is through MongoDb. I have isolated the MongoDb specific code using #.... at the start of the code
    section and ##.... at the end. Replace these code blocks with your own data storage code if necessary.

    The output of a function becomes the input for the following function.  Notice that as the number of symbols decreases
    as we progress through this class, the amount of data collected increases.

    Function List:

        -  stocks_list_under(amount)    -  Retrieves a complete list of stocks under a dollar(float) amount from gather_all_stock_symbols
        -  find_stock_correlation       -  Generates a list of correlated stocks pairs from stocks_list_under(amount)
        -  correlated_stock_data        -  Retrieves info and dividend data for each individual stock in the correlated stock pair
        -  correlated_stock_history     -  Retrieves historical data for the stocks from correlated_stock_data
        -  recommend_algorithm_symbols  -  Recommendation algorithm for the stocks. Returns 3 or better on a scale: 1-6
        -  compare_recommended_symbols  -  Compares the stock symbols in the corr_pair with those from recommend_algorithm_symbols
        -  find_best_stock_pair         -  Uses the yearly dividend payout / total stock prices ratio do determine best value

    """
    pass


today = datetime.today()
client = MongoClient( 'localhost', 27017 )


class StockSearchSupport():

    def get_info(self, symbol: str)->str:
        """
        Retrieves stock information from yfinance.
        Contains 122 elements of data.

        :param symbol: stock symbol
        """

        ticker = yf.Ticker( symbol: str)->str:
        return ticker.info
    

    def get_dividend(self, symbol):
        """
        Retrieves dividend information from yfinance.
        Contains 121 elements of data.

        :param symbol: stock symbol
        """

        ticker = yf.Ticker( symbol )
        return ticker.dividends
    

    @jit( nopython=True )
    def find_interval(self, var, type):
        """
        Calculates interval between dividend payments
        
        :param var: list of dividend dates
        """
        
        count = len( var ) - 3
        last_elem = var[count]
        freq_elem = var[count - 1]
        frequency = last_elem - freq_elem

        _temp = np.timedelta64( frequency, 'ns' )
        days = _temp.astype( 'timedelta64[D]' )
        interval = int( (days / np.timedelta64( 1, 'D' )) / 30 )
        return interval
    

    @jit( nopython=True )
    def interval_range(self, var):
        """
        Sometimes the interval between dividends are a little more, or less,
        than 90, 180 days because of weekends and holidays.
        If the interval falls between these ranges, the interval is
        assigned an integer
        """
        
        if var == 3:
            interval = 4
        elif var == 6:
            interval = 2
        else:
            interval = 1
        return interval
    

    def gather_all_stock_symbols():
        """
        Retrieves all available stock symbols from
        yahoo-fin.stock_info(si) and then removes duplicates.

        Total number of symbols = 9181.
        """

        # ......... MongoDb initialization................................
        db = client['trading']
        coll = db['all_stock_symbols']
        ##................................................................

        # Some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest.
        my_list = ['W', 'R', 'P', ]
        del_set = set()
        sav_set = set()

        # gather stock symbols from major US exchanges
        df1 = pd.DataFrame( si.tickers_sp500() )
        df2 = pd.DataFrame( si.tickers_nasdaq() )
        #        df3 = pd.DataFrame(si.tickers_dow())
        df4 = pd.DataFrame( si.tickers_other() )

        # convert DataFrame to list, then to sets
        sym1 = set( symbol for symbol in df1[0].values.tolist() )
        sym2 = set( symbol for symbol in df2[0].values.tolist() )
        #        sym3 = set( symbol for symbol in df3[0].values.tolist() )
        sym4 = set( symbol for symbol in df4[0].values.tolist() )

        # join the 4 sets into one. Because it's a set, there will be no duplicate symbols
        symbols = set.union( sym1, sym2, sym4 )  # sym3,

        for symbol in symbols:
            if len( symbol ) > 4 and symbol[-1] in my_list:  # we identify the above suffixed symbols here,
                del_set.add( symbol )  # and separate them here
            else:
                sav_set.add( symbol )  # updated data without suffixed symbols

        # insert data to MongoDb..........................................
        for symbol in sav_set:
            coll.insert_one( {"symbol": symbol} )
        ##................................................................

        print( f'Removed {len( del_set )} unqualified stock symbols...' )
        print( f'There are {len( sav_set )} qualified stock symbols...' )


class CorrelatedStockResearch(StockSearchSupport):
    """
    Class for determining stock pair correlation
    """

    def stocks_list_under(amount: int | float)->Union[int | float]:
        """
        Retrieves a complete list of stocks under a dollar amount in price
        from gather_all_stock_symbols(), then retrieves info and dividends data.

            Total number of stocks under 5 dollars = 893
            Number of those stocks with dividends = 128
            
        :param amount: Dollar amount of upper search limit

        ################## WARNING ##################
        
        This function takes about 8 hours to run on my system
        - Windows 10 Professional
        - AMD Ryzen 3800 CPU
        - NVIDIA 1660ti Graphics Card
        - 32 gb RAM
        """

        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['all_stock_symbols']
        col4 = db['div_stocks_under_5']  ## Change number if different self.amount
        col3 = db['stock_info']
        col2 = db['all_stocks_under_5_dollars']  ## Change this number, also
        data = col1.find( {}, {'_id': 0, 'symbol': 1} )
        ##................................................................

        datum = []

        for _d in data:
            for key, value in _d.items():
                try:
                    temp = get_info(value)
                    price = temp['open']  # find stock price

                    # .............. input stock.symbol, stock.info to MongoDb.........
                    col3.insert_one( {'symbol': value, 'info': temp} )
                    ##................................................................

                    if float( price ) < amount:
                        dividend = pd.DataFrame( get_dividend( value ) )  # get stock dividend info
                        dividend['index'] = dividend.index  # these next four lines formats the data for ease of use
                        dividend.index.name = None
                        dividend.rename( columns={"index": "Date"}, inplace=True )
                        dividend.reset_index( drop=True, inplace=True )

                        dataz = dividend.loc[:, 'Dividends']
                        for data in dataz:
                            datum.append( data )

                        # ...................insert symbol, price to MongoDb................
                        col2.insert_one( {'symbol': value, 'info': temp} )
                        ##.................................................................

                        time.sleep( .5 )  # we don't want to hog all the bandwidth

                        if len( dividend ) < 1:  # pass if no dividend payment
                            pass
                        else:
                            interval = find_interval( dividend['Date'], 'Date' )

                            last_divs = [x for x in dividend['Dividends']][count]  # find amount of last dividend paid
                            date_list = pd.DataFrame( [today - timedelta( days=x ) for x in
                            range( 370 )],
                                                      columns=['Date'] )  # list of all dates for past 370 days

                            for row in date_list['date']:
                                if str( last_divd )[:10] == str( row )[
                                                            :10]:  # if last dividend pay date is within the past year
                                    data = {  # dict formatted data for MongoDb
                                            'symbol'   : value,
                                            'last_date': last_divd,
                                            'last_divd': last_divs,
                                            'interval' : interval,
                                            'dividends': [datum],
                                    }
                                # .............................. insert data to MongoDb...............
                                    col4.insert_one( data )
                                ##...................................................................
                                else:
                                    pass
                    else:
                        pass
                except:
                    pass
                

    def find_stock_correlation():
        """
        Generates a list of 10 uncorrelated stocks pairs.  The closer the absolute value
        is to 0, the more uncorrelated the stock pairs are.

        Corrolated stocks move together (same or opposite direction). Uncorrolated stocks
        move independently of each other.

        Total number of uncorrelated stock pairs = 10
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['div_stocks_under_5']
        col2 = db['stock_pair_corellated']
        data = col1.find( {}, {'symbol': 1, '_id': 0} )
        ##................................................................

        syms = []

        for _d in data:
            syms.append( _d['symbol'] )

        start = today - timedelta( days=730 )  # collects two years data for correlation algorithm
        ticker = yf.Ticker( "SPY" )
        days = ticker.history( interval="1d", start=start, end=today ).shape[0]
        total = None

        for symbol in syms:
            ticker = yf.Ticker( symbol )
            close = ticker.history( interval="1d", start=start, end=today )[['Close']].pct_change( 1 )
            close.rename( columns={'Close': symbol}, inplace=True )

            if close.shape[0] < days:
                continue

            if total is None:
                total = close
            else:
                total = total.merge( close, left_index=True,
                                     right_index=True, )  # suffixes=("", "") <- this was originally in
                                                          # the total.merge(tuple). did not work on my
            corr_table = total.corr()                     # system. removed it and it worked
            corr_table['stock1'] = corr_table.index
            corr_table = corr_table.melt( id_vars='stock1', var_name="stock2" ).reset_index( drop=True )

            corr_table = corr_table[corr_table['stock1'] < corr_table['stock2']].dropna()
            corr_table['abs_value'] = np.abs( corr_table['value'] )
            highest_corr = corr_table.sort_values( "abs_value", ascending=True ).to_dict()

        df = pd.DataFrame( highest_corr )

        df.sort_values( by='abs_value', ascending=True, inplace=True )  # sort stock pairs in ascending absolute values
        my_stocks = df.head( 10 )                                       # Top ten uncorrelated stock pairs (closest to 0)

        # prepare data and insert to MongoDb...............................
        for row in my_stocks.iterrows():
            df1 = row[1]['stock1']
            df2 = row[1]['stock2']
            df3 = row[1]['abs_value']
            col2.insert_one( {'stock1': df1, 'stock2': df2, 'value': df3, } )
        ##.................................................................
        

    def correlated_stock_data():
        """
        Retrieves info and dividend data for each individual
        stock in the correlated stock_pair collection.

        Total number of symbols = 14
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col2 = db['stock_pair_corellated']
        col1 = db['corr_stock_info']
        data = col2.find( {}, {'_id': 0, 'stock1': 1, 'stock2': 1} )
        if col1:
            col1.drop()
        ##................................................................

        my_list = []
        my_set = set()

        for _d in data:
            my_set.add( _d['stock1'] )  # since an individual stock may be in more than one
            my_set.add( _d["stock2"] )  # stock pair, we will want to remove any duplicates

        for symbol in my_set:
            info = dict( get_info( symbol ) )      # gets stock info in dictionary format
            note = dict( get_dividend( symbol ) )  # gets stock dividend data in dictionary format
            plug = (symbol, info, note)            # assemble the data entry
            my_list.append( plug )                 # append the entry to my_list

        # create a dataframe for data massaging
        df = pd.DataFrame( my_list, columns=['symbol', 'info', 'dividend'] )

        my_divs = []

        for row in df.iterrows():
            dividend = row[1]['dividend']  # find the dividends (again in dictionary format)

            for key, value in dividend.items():
                data = {'date': key, 'amount': value}  # append all dividend dates and amounts for EACH symbol
                my_divs.append( data )

            symbol = row[1]['symbol']
            info = row[1]['info']

        # ...insert data to MongoDb...........................................

            col1.insert_one( {'symbol': symbol, 'info': info, 'dividend': [my_divs]} )
        ##...................................................................
        

    def correlated_stock_history(object):
        """
        Retrieves two years of history for each symbol in corr_stock_info,
        formats the data then updates the collection.
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['corr_stock_info']
        data = col1.find( {}, {'_id': 1, 'symbol': 1} )
        ##................................................................

        history = []

        for _d in data:
            symbol = _d['symbol']
            ticker = yf.Ticker( symbol )
            hist = pd.DataFrame( ticker.history( period='2y' ) )  # two year stock history

            hist['index'] = hist.index  # these four lines format the data for ease of use
            hist.index.name = None
            hist.rename( columns={"index": "Date"}, inplace=True )
            hist.reset_index( drop=True, inplace=True )
            hist.drop( columns=['Dividends', 'Stock Splits'], axis=1, inplace=True )
            hist = hist[:].values.tolist()

            for row in hist:
                date   = str( row[5] )
                open   = row[0]
                high   = row[1]
                low    = row[2]
                close  = row[3]
                volume = row[4]

                history.append( {
                        'date'  : date,
                        'open'  : open,
                        'high'  : high,
                        'low'   : low,
                        'close' : close,
                        'volume': volume
                } )

        # ...update data to MongoDb collection.............................
            col1.update_one( {'symbol': symbol},
                             {
                                     '$set': {'history': [history]}

                             }
                             )
        ##................................................................
        

    def recommend_algorithm_symbols():
        """
        Uses the 'gather_all_stock_symbols' list of symbols and retrieves
        the recommendation score (1-6) from finance.yahoo. Stocks
        with a score of 3 or better are saved.

        Total number of stocks with Recommendation Score greater than 3 = 3644

        May take up to 1 hour to run.
        """
        
        # ......... MongoDb initialization................................
        db = client["trading"]
        col1 = db['all_stock_symbols']
        col2 = db['AI_symbols_recommend']
        col3 = db['corr_symbol_recommendation']
        data = col1.find( {}, {'_id': 0, 'symbol': 1} )
        ##................................................................

        tickers = []
        recommendations = []

        for _d in data:
            tickers.append( _d['symbol'] )

        for ticker in tickers:
            lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
            rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
                      'modules=upgradeDowngradeHistory,recommendationTrend,' \
                      'financialData,earningsHistory,earningsTrend,industryTrend&' \
                      'corsDomain=finance.yahoo.com'

            url = lhs_url + ticker + rhs_url
            req = requests.get( url )
            if not req.ok:
                recommendation = 6
            try:
                result = req.json()['quoteSummary']['result'][0]
                recommendation = result['financialData']['recommendationMean']['fmt']
                data = ticker, recommendation
            except:
                recommendation = 6
                data = ticker, recommendation

            if float( recommendation ) < 3:  # change the integer for different recommendation scores
                recommendations.append( data )
            else:
                pass

        datum = dict( recommendations )
        for key, value in datum.items():
        # ...insert data to MongoDb.......................................
            col3.insert_one( {'symbol': key, 'recommend_value': value} )
        ##...............................................................
        

    def compare_recommended_symbols():
        """
        Compares the stock symbols in the correlated stock
        pair with those from recommend_algorithm_symbols().

        Total number of individual stocks from 14 (previous) in Recommended List = 4
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['corr_symbol_recommendation']
        col2 = db['corr_stock_info']
        col3 = db['corr_stock_in_recommended_list']
        data = col1.find( {}, {'_id': 0, 'symbol': 1, } )
        info = col2.find( {}, {'_id': 0, 'symbol': 1, } )
        ##................................................................

        data_list = []
        info_list = []

        for _d in data:
            for key, value in _d.items():
                data_list.append( value )

        for _i in info:
            for key, value in _i.items():
                info_list.append( value )

        for symbol in info_list:
            if symbol in data_list:
                # ..............................................................
                col3.insert_one( {'symbol': str( symbol )} )
            ##.............................................................
            else:
                print( f'{symbol} is not in the Recommendation List...' )
                

    def find_best_stock_pair(self):
        """
        Calculates the yearly dividend payout from stock symbols
        in the corr_symbol_recommendation data. That data is then
        compared with the price of the symbols in each stock pair
        and displayed by best value to worst value.

        May take an hour or more to run.
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['corr_symbol_recommendation']
        col2 = db['corr_stock_info']
        col3 = db['stock_pair_corellated']
        data = col1.find( {}, {'_id': 0, 'symbol': 1, } )
        ##................................................................

        data_list = []
        info_list = []
        mixed_bag = set()

        for _d in data:
            symbol = _d['symbol']
            # .....................................................................
            datum = col3.find( {}, {'stock1': 1, 'stock2': 1, '_id': 0} )
            ##....................................................................
            for _d in datum:
                dp1 = _d['stock1']  # retrieve the symbols for correlated stock Pairs
                dp2 = _d['stock2']
                # .....................................................................
                dp1_info = col2.find_one( {'symbol': dp1}, {'_id': 0} )  # get stock info for each symbol in stock pairs
                dp2_info = col2.find_one( {'symbol': dp2}, {'_id': 0} )
                ##....................................................................
                df1 = pd.Series( dp1_info )  # ['previousClose']         # get the latest closing price for each symbol
                df2 = pd.Series( dp2_info )

                close1 = df1['info']['previousClose']
                close2 = df2['info']['previousClose']

                ep1 = self.find_interval( df1['dividend'][0], 'date' )
                ep2 = self.find_interval( df2['dividend'][0], 'date' )

                count1 = len( df1 )
                count2 = len( df2 )

                last_divs1 = df1['dividend'][0][count1]
                last_divs2 = df2['dividend'][0][count2]

                next_divs1 = df1['dividend'][0][count1 - 1]
                next_divs2 = df2['dividend'][0][count2 - 1]

                interval1 = self.interval_range( ep1 )
                interval2 = self.interval_range( ep2 )

                avg_div1 = (float( last_divs1['amount'] ) + float(
                        next_divs1['amount'] )) / 2                 # usually, the dividend amount won't be the same each time
                avg_div2 = (float( last_divs2['amount'] ) + float(
                        next_divs2['amount'] )) / 2                 # therefore, I'm using the average of the two amounts

                yearly_div1 = ep1 * avg_div1  # total yearly dividend payout = frequency of
                yearly_div2 = ep2 * avg_div2  # dividends * the average dividend amount

                divs = yearly_div1 + yearly_div2  # add yearly dividend amount of both symbols in stock pair
                cloz = close1 + close2            # add closing price of both symbols in stock pair

                mixed_bag.add(
                        (dp1,
                         close1,
                         dp2,
                         close2,
                         cloz,
                         divs,
                         divs / cloz,)
                )  # tie it all up nicely

        df = pd.DataFrame( mixed_bag, columns=['stock1',
                                               'price1',
                                               'stock2',
                                               'price2',
                                               'tot_price',
                                               'tot_dividends',
                                               'd/p ratio',
                                               ]
                           )

        df.sort_values( by='d/p ratio', ascending=False,
                        inplace=True )             # sort by column dividend/price ratio from low to high. The lower
        pd.set_option( 'display.max_rows', None )  # the ratio, the better the bang for your buck
        pd.set_option( 'display.max_columns', None )
        pd.set_option( 'display.width', 1000 )
        pd.set_option( 'display.colheader_justify', 'center' )
        pd.set_option( 'display.precision', 4 )

        print()
        print( "d/p ratio = tot_dividends divided by tot_price" )
        print( "Sorted by best to worst value for your viewing pleasure." )
        print()
        print( df )
        

    def chosen_history(symbol: str)->str:
        """
        Retrieves stock symbol history data from
        the database then plots to graph.
        
        :param symbol: Stock symbol
        """
        
        # ......... MongoDb initialization................................
        db = client['trading']
        col1 = db['corr_stock_info']
        data = col1.find( {}, {'_id': 0, 'symbol': 1, 'history': 1} )

        datum = []

        for _d in data:

            for elem in _d['history'][0]:
                datum.append( [elem['date'][:10], elem['close']] )

            df = pd.DataFrame( datum, columns=['date', 'close'] ).tail( 365 )
            df.plot( x='date', y='close', kind='line' )
            plt.show()


if __name__ == '__main__':
    main()
