import pandas as pd
import datetime as dt
from dateutil import parser

dataset = pd.read_csv('yahoo_dataset_gspc.csv')
dataset = dataset.dropna()

# Diccionario para traducir meses en español a inglés
MESES_ESP = {
    "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr", "may": "May", "jun": "Jun",
    "jul": "Jul", "ago": "Aug", "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec"
}

def corregir_fecha(fecha_str):
    try:
        partes = fecha_str.split()
        if len(partes) == 3:  # Asegurar que tiene día, mes y año
            dia, mes, anio = partes
            mes = MESES_ESP.get(mes.lower(), mes)  # Traducir mes al inglés
            fecha_corregida = f"{dia} {mes} {anio}"
            fecha_dt = parser.parse(fecha_corregida, dayfirst=True)  # Convertir a datetime
            return fecha_dt.strftime("%Y-%m-%d")  # Devolver en formato legible
    except Exception as e:
        print(f"Error con la fecha: {fecha_str} - {e}")
        return None  # Retorna None si hay un error

def actualizar_fecha(df, columna):
    df[columna] = df[columna].apply(lambda x: corregir_fecha(x) if pd.notna(x) else None)
    return df

dataset = actualizar_fecha(dataset, "Date")
dataset = dataset.sort_values("Date")

def calculateFields(dataset):
    if 'Date' not in dataset:
        dataset['Date'] = dataset['date']
        dataset = dataset.drop(columns=["date"])
    if 'Adj Close' in dataset:
        dataset = dataset.drop(columns=['Adj Close'])
    if 'Adj_close' in dataset:
        dataset = dataset.drop(columns=['Adj_close'])
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset["DateOrdinal"] = dataset["Date"].dt.strftime("%Y%m%d").astype(int)
    #dataset['DateOrdinal'] = dataset['Date'].map(dt.datetime.toordinal)
    dataset['average_price'] = dataset[['Low', 'High']].mean(axis=1)
    dataset = dataset.drop(columns=["Volume", "Close", "Open", "Low", "High"])
    dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
    dataset['DayOfYear'] = dataset['Date'].dt.dayofyear
    dataset['WeekOfYear'] = dataset['Date'].dt.isocalendar().week

    dataset['is_start_of_month'] = (dataset['Date'].dt.day == 1).astype(int)
    dataset['is_end_of_month'] = (dataset['Date'] == dataset['Date'] + pd.offsets.MonthEnd(0)).astype(int)
    dataset['quarter'] = dataset['Date'].dt.quarter
    dataset['SerieNumber'] = (dataset['Date'] - dataset['Date'].min()).dt.days + 1

    dataset = dataset.dropna().reset_index(drop=True)
    if 'Symbol' in dataset:
        dataset['Symbol'] = pd.factorize(dataset.Symbol)[0]
    else:
        dataset['Symbol'] = pd.factorize(dataset.Ticker)[0]
        dataset = dataset.drop(columns=['Ticker'])

    return dataset


dataset = calculateFields(dataset)

dataset['Date'] = pd.to_datetime(dataset['Date'])

dataset.set_index('Date', inplace=True)

# Cambio diario
dataset['Daily_Change'] = dataset['average_price'].diff()
dataset['Daily_Change_Pct'] = dataset['average_price'].pct_change()

# Día al alza o a la baja
dataset['Up'] = dataset['Daily_Change'] > 0
dataset['Down'] = dataset['Daily_Change'] < 0

# Conteo de días al alza o a la baja en determinado rango
dataset['Up_Last_30'] = dataset['Up'].rolling('30D', closed='left').sum()
dataset['Down_Last_30'] = dataset['Down'].rolling('30D', closed='left').sum()
dataset['Up_Last_7'] = dataset['Up'].rolling('7D', closed='left').sum()
dataset['Down_Last_7'] = dataset['Down'].rolling('7D', closed='left').sum()
dataset['Up_Last_14'] = dataset['Up'].rolling('14D', closed='left').sum()
dataset['Down_Last_14'] = dataset['Down'].rolling('14D', closed='left').sum()

# Cálculo de rachas consecutivas
def compute_streaks(series):
    streaks = []
    count = 0
    prev = None
    for val in series:
        if val == prev and val:
            count += 1
        else:
            count = 1 if val else 0
        prev = val
        streaks.append(count)
    return streaks

# Cantidad de días seguidos que va al alza o a la baja
dataset['Up_Streak'] = compute_streaks(dataset['Up'])
dataset['Down_Streak'] = compute_streaks(dataset['Down'])

# Racha máxima en los últimos 30 días
dataset['Max_Up_Streak_30'] = dataset['Up_Streak'].rolling('30D', closed='left').max()
dataset['Max_Down_Streak_30'] = dataset['Down_Streak'].rolling('30D', closed='left').max()

# media movil

dataset['SMA_30'] = dataset['average_price'].rolling(window=30, closed='left').mean()

# disparity index

dataset['Disparity_SMA_30'] = ((dataset['average_price'].shift(1) - dataset['SMA_30']) / dataset['SMA_30']) * 100

# EMAs Exponential Moving Average

dataset['EMA_30'] = dataset['average_price'].shift(1).ewm(span=30, adjust=False).mean()

dataset.dropna(inplace=True)

pivote_date = dt.datetime.strptime("01/01/2018", "%d/%m/%Y")
max_day = dt.datetime.strptime("31/12/2020", "%d/%m/%Y")
dataset.reset_index(inplace=True)
dataset = dataset[(dataset['Date'] >= pivote_date) & (dataset['Date'] <= max_day)]

dataset.to_csv('sp500_enriquecido.csv', index=False)