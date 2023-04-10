import pandas as pd

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
print(Y)

