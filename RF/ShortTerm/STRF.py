import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv(r'raw data/daily/combined/combined.csv')
X = pd.DataFrame()
for i in range(1,len(df)):
    if i < 3:
        pass
    else:
        tempA = pd.DataFrame(df.loc[[i-1]])
        tempA.columns = [str(col) + '_1d_ago' for col in tempA.columns]
        tempB = pd.DataFrame(df.loc[[i-2]])
        tempB.columns = [str(col) + '_2d_ago' for col in tempB.columns]
        tempC = pd.DataFrame(df.loc[[i-3]])
        tempC.columns = [str(col) + '_3d_ago' for col in tempC.columns]
        tempA = tempA.set_axis([i], copy=True)
        tempB = tempB.set_axis([i], copy=True)
        tempC = tempC.set_axis([i], copy=True)
        concat = pd.concat([tempA,tempB,tempC], axis="columns")
        if i == 3:
            X = concat
        else:
            X = pd.concat([X, concat], ignore_index=True)
Y = df['T_DAILY_AVG']
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
X = X[['T_DAILY_AVG_1d_ago',
'P_DAILY_CALC_1d_ago',
'SOLARAD_DAILY_1d_ago',
'SUR_TEMP_DAILY_AVG_1d_ago',
# 'RH_DAILY_AVG_1d_ago',
# 'SOIL_MOISTURE_5_DAILY_1d_ago',
# 'SOIL_TEMP_5_DAILY_1d_ago',
'T_DAILY_AVG_2d_ago',
'P_DAILY_CALC_2d_ago',
'SOLARAD_DAILY_2d_ago',
'SUR_TEMP_DAILY_AVG_2d_ago',
# 'RH_DAILY_AVG_2d_ago',
# 'SOIL_MOISTURE_5_DAILY_2d_ago',
# 'SOIL_TEMP_5_DAILY_2d_ago',
'T_DAILY_AVG_3d_ago',
'P_DAILY_CALC_3d_ago',
'SOLARAD_DAILY_3d_ago',
'SUR_TEMP_DAILY_AVG_3d_ago'
# 'RH_DAILY_AVG_3d_ago',
# 'SOIL_MOISTURE_5_DAILY_3d_ago',
# 'SOIL_TEMP_5_DAILY_3d_ago'
]]
x_train = X.iloc[:1063,:]
x_test = X.iloc[1063:,:]
y_train = Y.iloc[:1063]
y_test = Y.iloc[1063:]
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(x_train, y_train)
print(regr.score(x_test, y_test))
