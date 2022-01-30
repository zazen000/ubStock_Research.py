# ubStock_Research.py
Another Stock Evaluation Module

    ubStock_Research.py
    Utility_Belt Designs, Tacoma, WA
    www.pythonOver60.rocks
    @author: ZennDogg, with a lot of help for find_stock_correlation from outside sources
    zenndogg@outlook.com
    python 3.10
    
    
    class StockSearchSupport: Contains support functions for class CorrelatedStockResearch.
    
    Function List:
    
        -  get_info                     -  Retrieves stock information from yfinance. Used by several functions
        -  get_dividend                 -  Retrieves dividend information from yfinance. Used by several functions
        -  find_interval                -  Determines the interval between dividend payments (in months)
        -  interval_range               -  Transforms the return of find_interval into a single digit integer
        -  gather_all_stock_symbols     -  Finds stock symbols from the Dow, SP500, NASDAQ and others. Removes all duplicates

    
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
        -  find_best_stock_pair         -  Uses the yearly dividend payout divided by total stock prices ratio to determine best value
       
