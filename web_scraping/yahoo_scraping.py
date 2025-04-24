import time
import pandas as pd
from web_scraping.PageDownloaderHandler import PageDownloadHandler

def getByTicker(ticker):
    pageDownloadHandler = PageDownloadHandler()
    url = f"https://es-us.finanzas.yahoo.com/quote/{ticker}/history/?period1=0&period2=1733268939"
    try:
        return pageDownloadHandler.download(url)
    except:
        time.sleep(60)
        return pageDownloadHandler.download(url)

def appendToFile(dataDict, fileName, ticker):
    f = open(fileName, 'a')
    for key in dataDict:
        strData = []
        for i in range(6):
            strData.append(f"{dataDict[key][i]}".replace(",", ""))
        dataDict[key] = strData
        f.write(f"{key},{ticker},{dataDict[key][0]},{dataDict[key][1]},{dataDict[key][2]},{dataDict[key][3]},{dataDict[key][4]},{dataDict[key][5]}\n")
    f.close()

f = open("yahoo_dataset_gspc.csv", "w")
f.write("Date,Ticker,Open,High,Low,Close,Adj_close,Volume\n")
f.close()
f0 = open("yahoo_void_spy.csv", "w")
dataset = pd.read_csv('../datasets/sp_500_stocks/sp500_companies.csv')
#for index, row in dataset.iterrows():
symbol = "%5EGSPC"  #"SPY" # row["Symbol"]
history = getByTicker(symbol)
if history:
    appendToFile(history, "yahoo_dataset_gspc.csv", symbol)
else:
    f0.write(f"{symbol}\n")
#print(f"Procesado: {symbol} {int(index) + 1} de 500")
#time.sleep(5)
f0.close()