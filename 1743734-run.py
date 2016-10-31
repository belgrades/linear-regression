import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gendata(x):
    m, p = x.shape
    df = np.ones((m, p+1))
    df[:, 1:] = x
    return df


def preprocessing(df, objective, columns):
    df = df[columns]
    m_class = df[objective]
    columns.remove(objective)
    df = df[columns].apply(lambda x: (x - x.mean()) / x.std())
    df[objective] = m_class
    return df


def genplot(y, x, w):
    fig, x_min, x_max = plt.figure(), x.min(), x.max()
    #fig.x('Average Growing Season Temperature')
    #fig.ylabel('Price')
    #fig.suptitle('The Wine Equation')

    eps=1.0
    plt.plot([x_min-eps, x_max+eps], [w[1] * (x_min-eps) + w[0], w[1] * (x_max+eps)+ w[0]], 'b-', lw=1)
    plt.plot(x, y, 'ro')
    plt.xlim([x_min-eps, x_max+eps])
    plt.xlabel('Average Growing Season Temperature')
    plt.ylabel('log(Price)')
    plt.suptitle('The Wine Equation')
    plt.grid()
    fig.savefig('1743734.PNG', dpi=fig.dpi)

def main():
    lib = importlib.import_module("1743734-lib")
    df = pd.read_csv('wine.csv')
    df = preprocessing(df=df,
                       objective='Price',
                       columns=['Price', 'WinterRain', 'AGST', 'HarvestRain', 'Age'])

    y = np.array(df['Price'].values)
    x = gendata(df.as_matrix(columns=['AGST']))

    w = lib.descent(y=y,
                    x=x,
                    alpha=1e-2,
                    itr=1e4,
                    eps=1e-3)
    print(w)
    print(lib.r2(y=y, c=x, x=w))

    genplot(y=y, x=np.array(df['AGST'].values), w=w)

    x = gendata(df.as_matrix(columns=['AGST', 'WinterRain', 'HarvestRain', 'Age']))

    w = lib.descent(y=np.array(df['Price'].values),
                    x=x,
                    alpha=1e-2,
                    itr=1e4,
                    eps=1e-3)
    print(w)
    print(lib.r2(y=y, c=x, x=w))

if __name__ == '__main__':
    main()