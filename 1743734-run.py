import importlib
import numpy as np
import pandas as pd


def preprocessing(df):
    columns = ['Price', 'WinterRain', 'AGST', 'HarvestRain', 'Age']
    df = df[columns]
    price = df['Price']
    columns.remove('Price')
    df = df[columns].apply(lambda x: (x - x.mean()) / x.std())
    df['Price'] = price

    return(df)


def main():
    lib = importlib.import_module("1743734-lib")
    df = pd.read_csv('wine.csv')
    df = preprocessing(df)
    col = df.columns.tolist()
    col.remove('Price')
    df[col].apply(lambda x: (x - x.mean()) / x.std())
    print(df.head())

    a, b = 2.0, 4.0

    x, y = lib.gendata(np.array([b, a]), 200)
    print(lib.descent(y, x, 0.0001, 100))
    lib.genplot(x, y, a=a, b=b)

if __name__ == '__main__':
    main()