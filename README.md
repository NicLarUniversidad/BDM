# Trabajo final de Bases de Datos Masivas

## Organización del proyecto

1. En la carpeta raíz, se tienen dos versiones de Random Forest for Time Series (RFTS). Una de regresión y otra de clasificación.

2. En la carpeta test se tienen 3 pruebas. Una del rendimiento general de RFTS contra Random Forest (RF), utilizando el dataset S&P 500 Stocks (Daily updated).

3. En la carpeta web_scraping hay dos scrapers para obtener datos de los sitios Yahoo Finanzas y Finviz. Ambos leen información publicada en sus sitios públicos, parsean el HTML y guardan datos en formato CSV.


## Pruebas

1. Primero se probó utilizando todos los datos, usando un 80% como conjunto de entrenamiento y el 20% más acutalizado como conjunto de pruebas. Nombre de archivo: test001_cajas.ipynb.

2. Segundo se probó utilizando un rango de entrenamiento de un año.

3. Con la tercera prueba se utilizaron ventanas deslizantes de 30 y 60 días. Nombre de archivo: test003_ventana_deslizante.ipynb.

## Requerimientos

Además de Python y las librerías de Pandas, Sklearn.
Se requiere descargar el dataset https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/data.
Guardar los archivo en el sigueinte directorio.
        
        datasets/sp_500_stocks

También se puede guardar en otra carpeta pero se debe configurar las notebooks en las primeras líneas, así buscar el archivo sp500_index.csv correctamente.
