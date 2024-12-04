import time
import pandas as pd
from web_scraping.PageDownloaderHandler import PageDownloadHandler

def getByTicker(ticker):
    pageDownloadHandler = PageDownloadHandler()
    url = f"https://es-us.finanzas.yahoo.com/quote/{ticker}/history/?period1=0&period2=1733268939"
    history0 = None
    try:
        history0 = pageDownloadHandler.download(url)
    except:
        time.sleep(60)
        history0 = pageDownloadHandler.download(url)
    return history0

def appendToFile(dataDict, fileName, ticker):
    f = open(fileName, 'a')
    for key in dataDict:
        f.write(f"{key},{ticker},{dataDict[key][0]},{dataDict[key][1]},{dataDict[key][2]},{dataDict[key][3]},{dataDict[key][4]}\n")
    f.close()

f = open("yahoo_dataset.csv", "w")
f.write("date,ticker,open,high,low,close,adj_close\n")
f.close()
f0 = open("yahoo_void.csv", "w")
dataset = pd.read_csv('../datasets/sp_500_stocks/sp500_companies.csv')
for index, row in dataset.iterrows():
    symbol = row["Symbol"]
    history = getByTicker(symbol)
    if history:
        appendToFile(history, "yahoo_dataset.csv", symbol)
    else:
        f0.write(f"{symbol}\n")
    print(f"Procesado: {symbol} {index} de 500")
    time.sleep(5)
f0.close()