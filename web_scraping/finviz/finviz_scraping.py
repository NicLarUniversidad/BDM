import json
import time
import pandas as pd
import requests

def scrap_by_ticker(ticker):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0'}
    url = f"https://finviz.com/api/quote.ashx?aftermarket=false&events=false&financialAttachments=&instrument=stock&patterns=false&premarket=false&rev=1732894131518&ticker={ticker}&timeframe=d&type=new"
    page = requests.get(url, headers=headers)
    jsonData = json.loads(page.content)
    volumes = jsonData['volume']
    dates = jsonData['date']
    opens = jsonData['open']
    closes = jsonData['close']
    highs = jsonData['high']
    lows = jsonData['low']
    f0 = open("finviz_dataset_spy.csv" ,"a")
    for i in range (len(volumes)):
        f0.write(f"{dates[i]},{ticker},{opens[i]},{closes[i]},{highs[i]},{lows[i]},{volumes[i]}\n")
    f0.close()

f = open("finviz_dataset_spy.csv" ,"w")
f.write("Date,Ticker,Open,Close,High,Low,Volume\n")
f.close()
dataset = pd.read_csv('../../datasets/sp_500_stocks/sp500_companies.csv')
#for index, row in dataset.iterrows():
symbol = "SPY" #row["Symbol"]
scrap_by_ticker(symbol)
#print(f"Procesado: {symbol} {index} de 500")
#time.sleep(5)