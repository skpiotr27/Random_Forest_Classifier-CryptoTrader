import pandas as pd
import pandas_ta as ta
import numpy as np
import talib 

def get_df(klines):
    # Convert the data to a pandas DataFrame object.
    df = pd.DataFrame(klines, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                      'Close_time', 'quote_asset_Volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Transforming the received data into OHCLV and adding technical analysis indicators.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df["RSI"] = ta.rsi(df["Close"], 14)
    df["SMA_1"] = ta.sma(df["Close"], 5)
    df["SMA_2"] = ta.sma(df["Close"], 5)
    df["SMA_3"] = ta.sma(df["Close"], 30)
    df["SMA_4"] = ta.sma(df["Close"], 60)
    df["AO"] = ta.ao(df["High"], df["Low"], 5, 34)
    df["APO"] = ta.apo(df["Close"], 12, 34)
    df["BIAS"] = ta.sma(df["Close"], 26)
    df["BOP"] = ta.bop(df["Open"], df["High"], df["Low"], df["Close"])
    df["BRAR"] = ta.bop(df["Open"], df["High"], df["Low"], df["Close"], 26)
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], 14)
    df["CFO"] = ta.cfo(df["Close"], 9)
    df["CG"] = ta.cg(df["Close"], 10)
    df["CMO"] = ta.cmo(df["Close"], 100)
    df["COPP"] = ta.coppock(df["Close"], 10, 11, 14)
    df["INERIA"] = ta.inertia(open=df["Open"], high=df["High"],
                              low=df["Low"], close=df["Close"], length=20, rvi_length=14)
    df["STOCHRSI_k"] = ta.stochrsi(df["Close"], length=14, rsi_length=14, k=3, d=3)[
        "STOCHRSIk_14_14_3_3"]
    df["STOCHRSI_d"] = ta.stochrsi(df["Close"], length=14, rsi_length=14, k=3, d=3)[
        "STOCHRSId_14_14_3_3"]
    df["RSX"] = ta.rsx(df["Close"], 14)
    df["STOCH_k"] = ta.stoch(df["High"], df["Low"], df["Close"],
                             length=14, rsi_length=14, k=3, d=3)["STOCHk_3_3_3"]
    df["STOCH_d"] = ta.stoch(df["High"], df["Low"], df["Close"],
                             length=14, rsi_length=14, k=3, d=3)["STOCHd_3_3_3"]
    df["MOM"] = ta.mom(df["Close"], 1)
    df["PSL"] = ta.psl(df["Close"], df["Open"], 12)
    df["MACD"] = ta.macd(df["Close"], 12, 26)["MACD_12_26_9"]
    df["MACD_h"] = ta.macd(df["Close"], 12, 26)["MACDh_12_26_9"]
    df["MACD_s"] = ta.macd(df["Close"], 12, 26)["MACDs_12_26_9"]
    mf = df.ta.cdl_pattern(name="all")
    df = pd.concat([df, mf], axis=1)
    df["EBSW"] = ta.ebsw(df["Close"], 14)
    df["TTM"] = ta.ttm_trend(df["High"], df["Low"], df["Close"], 6)
    dm = ta.dm(df["High"], df["Low"])
    df = pd.concat([df, dm], axis=1)
    eri = ta.eri(df["High"], df["Low"], df["Close"], 14)
    df = pd.concat([df, eri], axis=1)
    fisher = ta.fisher(df["High"], df["Low"], 9, 1)
    df = pd.concat([df, fisher], axis=1)
    kdj = ta.kdj(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, kdj], axis=1)
    kst = ta.kst(df["Close"])
    df = pd.concat([df, kst], axis=1)
    pgo = ta.pgo(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, pgo], axis=1)
    ppo = ta.ppo(df["Close"])
    df = pd.concat([df, ppo], axis=1)
    pvo = ta.pvo(df["Volume"])
    df = pd.concat([df, pvo], axis=1)
    roc = ta.roc(df["Close"])
    df = pd.concat([df, roc], axis=1)
    rvgi = ta.rvgi(df["Open"], df["High"], df["Low"], df["Close"])
    df = pd.concat([df, rvgi], axis=1)
    stc = ta.stc(df["Close"])
    df = pd.concat([df, stc], axis=1)
    slope = ta.slope(df["Close"])
    df = pd.concat([df, slope], axis=1)
    smi = ta.smi(df["Close"])
    df = pd.concat([df, smi], axis=1)
    squeeze = ta.squeeze(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, squeeze], axis=1)
    squeeze_pro = ta.squeeze_pro(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, squeeze_pro], axis=1)
    
    df["RSI_Yesterday"] = df["RSI"].shift(1)
    df["RSI_Yesterday"] = (df["RSI"]/df['RSI_Yesterday'])

    # Adding to the columns from the list subsequent columns with the value of the column/value of the column from the previous period.
    columns = ["SMA_1", "SMA_2", "SMA_3", "SMA_4", "AO", "APO", "BIAS", "CG", "CMO", "COPP",
             "INERIA", "STOCHRSI_k", "STOCHRSI_d", "STOCH_k", "STOCH_d",  "MACD", "MACD_h", "MACD_s"]

    for column in columns:
        df[column + "_Yesterday"] = df[column].shift(1)
        df[column + "_Yesterday"] = df[column] / df[column + "_Yesterday"]
        
    df["ANGLE"] = talib.LINEARREG_ANGLE(df["Close"])
    df['Value'] = (df['ANGLE'].diff() > 0).astype(int)
    # Adding a target column - 1 if the next closing price will be higher than the current one - green candle, 0 if the next closing price will be lower than the current one - red candle.
    
    df["Tommorow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tommorow"] > df["Close"]).astype(int)
    
    # Changing inf values to 1 or -1 and removing NaN values.
    df = df.replace({-np.inf: -1, np.inf: 1})
    df = df.dropna()

    return df
