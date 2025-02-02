import time
from web_scraping.PageDownloaderHandler import PageDownloadHandler

def getByTicker(ticker):
    pageDownloadHandler = PageDownloadHandler()
    url = f"https://es-us.finanzas.yahoo.com/quote/{ticker}/history/?period1=0&period2=1733268939"
    history0 = pageDownloadHandler.download(url)
    return history0

def appendToFile(dataDict, fileName, ticker):
    f = open(fileName, 'a')
    for key in dataDict:
        f.write(f"{key},{ticker},{dataDict[key][0]},{dataDict[key][1]},{dataDict[key][2]},{dataDict[key][3]},{dataDict[key][4]},{dataDict[key][5]}\n")
    f.close()

f = open("yahoo_dataset_missed.csv", "w")
f.write("date,ticker,open,high,low,close,adj_close\n")
f.close()
f0 = open("yahoo_void.csv", "r")
lines = f0.readlines()
for symbol in lines:
    history = getByTicker(symbol)
    if history:
        appendToFile(history, "yahoo_dataset_missed.csv", symbol)
    else:
        print(f"Can't download {symbol}")
    print(f"Procesado: {symbol}")
    time.sleep(5)
f0.close()