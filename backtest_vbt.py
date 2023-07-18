import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import vectorbt as vbt
import plotly.io as pio


# Download data from csv.
df = pd.read_csv('data_set.csv')
df.set_index('Index', inplace=True)

# The predictors that we will use to train the model.
predictors = ["RSI", "SMA_1", "SMA_2", "SMA_3", "SMA_4", "AO", "APO", "BIAS", "BOP", 
              "BRAR", "CCI", "CFO", "CG", "CMO", "COPP", "INERIA", "STOCHRSI_k", "STOCHRSI_d", "RSX", "STOCH_k", "STOCH_d", "MOM", "PSL", "MACD", "MACD_h", 
              "MACD_s", "CDL_3INSIDE", "CDL_3OUTSIDE",  
               "CDL_BELTHOLD",  "CDL_CLOSINGMARUBOZU",  "CDL_DARKCLOUDCOVER", "CDL_DOJI_10_0.1", 
              "CDL_DOJISTAR", "CDL_DRAGONFLYDOJI", "CDL_ENGULFING",  "CDL_GRAVESTONEDOJI", "CDL_HAMMER", "CDL_HANGINGMAN", 
              "CDL_HARAMI", "CDL_HARAMICROSS", "CDL_HIGHWAVE", "CDL_HIKKAKE",  "CDL_HOMINGPIGEON", "CDL_INSIDE", "CDL_INVERTEDHAMMER", 
                "CDL_LONGLEGGEDDOJI", "CDL_LONGLINE", "CDL_MARUBOZU",  
               "CDL_PIERCING", "CDL_RICKSHAWMAN",   "CDL_SHOOTINGSTAR", "CDL_SHORTLINE", "CDL_SPINNINGTOP", 
                "CDL_TAKURI", "CDL_TASUKIGAP", "CDL_XSIDEGAP3METHODS",
              "EBSW", "TTM", "DMP_14", "DMN_14", "BULLP_14", "BEARP_14", "FISHERT_9_1", "FISHERTs_9_1", "K_9_3", "D_9_3", "J_9_3", "KST_10_15_20_30_10_10_10_15", "KSTs_9", "PGO_14",
              "PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9", "PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9", "ROC_10", "RVGI_14_4", "RVGIs_14_4", "STC_10_12_26_0.5", "STCmacd_10_12_26_0.5",
              "STCstoch_10_12_26_0.5", "SLOPE_1", "SMI_5_20_5", "SMIs_5_20_5", "SMIo_5_20_5", "SQZ_20_2.0_20_1.5", "SQZ_ON", "SQZPRO_20_2.0_20_2_1.5_1", 
              "SQZPRO_ON_WIDE", "SQZPRO_ON_NORMAL",  "SQZPRO_OFF",  "RSI_Yesterday", "SMA_1_Yesterday", "SMA_2_Yesterday", 
              "SMA_3_Yesterday", "SMA_4_Yesterday", "AO_Yesterday", "APO_Yesterday", "BIAS_Yesterday", 
              "CG_Yesterday", "CMO_Yesterday", "COPP_Yesterday",  "INERIA_Yesterday", "STOCHRSI_k_Yesterday", "STOCHRSI_d_Yesterday", 
              "STOCH_k_Yesterday", "STOCH_d_Yesterday", "MACD_Yesterday", "MACD_h_Yesterday", "MACD_s_Yesterday"]

# Load model
model = joblib.load('model.pkl')

# Generating predictions for the Target column.
df["preds"] = model.predict(df[predictors])

# Evaluating the accuracy of the model.
accuracy = accuracy_score(df["Target"], df["preds"])
print("Accuracy model:", accuracy)

# Copy value of df["Close"] to price
price = df[["Close"]].copy()

# Generating entries and exits based on the state of Target.
entries = df["preds"]==1
exits = df["preds"]==0

# Creating a  vbt portfolio from the entries and exits signals.
pf=vbt.Portfolio.from_signals(price["Close"],entries,exits, sl_stop = 0.01)
# Print stats of trading
print(pf.stats())

# Generating a plot.
fig = pf.plot()

# Save and show plot
pio.write_html(fig,file = "trading_chart.html", auto_open=True)