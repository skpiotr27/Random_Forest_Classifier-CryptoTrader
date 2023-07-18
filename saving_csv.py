from binance.client import Client
from download_data import get_df
from decouple import config

# Creating an instance of the client with keys from the env file.
client = Client(config("API_KEY"), config("SECRET_KEY"))

# Setting the data to download.
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
start_time = '2023-04-01 00:00:00'
end_time = '2023-05-01 00:00:00'

# Fetching data from the Binance API.
klines = client.get_historical_klines(symbol, interval, start_time, end_time)

# Invoking the get_df function that formats the df and adds AT indicators.
df = get_df(klines)

# Saving the file to csv.
df.to_csv('data_set.csv', index_label='Index')