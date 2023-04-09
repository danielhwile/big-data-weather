import pandas as pd
from sklearn.neural_network import MLPRegressor
df = pd.read_csv(r'raw data/Monthly/CRNM0102-NC_Durham_11_W.csv')
x = df[['LST_MO','P_MONTHLY_CALC','SOLRAD_MONTHLY_AVG']]
y = df['T_MONTHLY_AVG']
x_train = x.iloc[:152,:]
x_test = x.iloc[152:,:]
y_train = y.iloc[:152]
y_test = y.iloc[152:]
regr = MLPRegressor(hidden_layer_sizes=500,activation='logistic',solver='lbfgs',random_state=1, learning_rate='adaptive', max_iter=1000).fit(x_train, y_train)
predictions = regr.predict(x_test)
print(predictions)
# print(y_test[:5])
print(regr.score(x_test, y_test))