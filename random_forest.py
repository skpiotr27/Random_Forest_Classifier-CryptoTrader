from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Download data from csv.
df = pd.read_csv('data_set.csv')
df.set_index('Index', inplace=True)

# Print data.
print(df)

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
              

X = df[predictors]
y = df["Target"]

# Splitting the dataset into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Create model RFC
model  = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=10,min_samples_leaf=20,verbose=1,warm_start=True,n_jobs=-1,random_state=42,min_impurity_decrease=0,criterion="entropy")

# Training the model and generating predictions for the test and training sets.
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

y_pred_2 = model.predict(X_train)
accuracy_2 = accuracy_score(y_train,y_pred_2)

# Generating a list of features.
importances = model.feature_importances_

# Creating a dictionary with feature names and their importances.
feature_importances = {}
for feature_name, importance in zip(X_train.columns, importances):
    feature_importances[feature_name] = importance

# Sorting features by importance.
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

# Print the sorted features.
for feature_name, importance in sorted_features:
    print(f"{feature_name}: {importance}") 

# Save the model to a file.
joblib.dump(model, 'model.pkl')

# Print the accuracy of train and test sets. 
print("Accuracy [TRAIN]:", accuracy_2)
print("Accuracy [TEST]:", accuracy)