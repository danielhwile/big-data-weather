import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv(r'raw data/Monthly/CRNM0102-NC_Durham_11_W.csv')
X = pd.DataFrame()
for i in range(1,len(df)):
    if i < 3:
        pass
    else:
        tempA = pd.DataFrame(df.loc[[i-1]])
        tempA.columns = [str(col) + '_1m_ago' for col in tempA.columns]
        tempB = pd.DataFrame(df.loc[[i-2]])
        tempB.columns = [str(col) + '_2m_ago' for col in tempB.columns]
        tempC = pd.DataFrame(df.loc[[i-3]])
        tempC.columns = [str(col) + '_3m_ago' for col in tempC.columns]
        tempA = tempA.set_axis([i], copy=True)
        tempB = tempB.set_axis([i], copy=True)
        tempC = tempC.set_axis([i], copy=True)
        concat = pd.concat([tempA,tempB,tempC], axis="columns")
        if i == 3:
            X = concat
        else:
            X = pd.concat([X, concat], ignore_index=True)
Y = df['T_MONTHLY_AVG']
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
Y = Y.reindex(Y.index.drop(0)).reset_index(drop=True)
X = X[['LST_MO_1m_ago','T_MONTHLY_AVG_1m_ago',
'P_MONTHLY_CALC_1m_ago',
'SOLRAD_MONTHLY_AVG_1m_ago',
'SUR_TEMP_MONTHLY_AVG_1m_ago',
'T_MONTHLY_AVG_2m_ago',
'P_MONTHLY_CALC_2m_ago',
'SOLRAD_MONTHLY_AVG_2m_ago',
'SUR_TEMP_MONTHLY_AVG_2m_ago',
'T_MONTHLY_AVG_3m_ago',
'P_MONTHLY_CALC_3m_ago',
'SOLRAD_MONTHLY_AVG_3m_ago',
'SUR_TEMP_MONTHLY_AVG_3m_ago'
]]
x_train = X.iloc[:152,:]
x_test = X.iloc[152:,:]
y_train = Y.iloc[:152]
y_test = Y.iloc[152:]
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(x_train, y_train)
print(regr.score(x_test, y_test))