import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gendata(x):
    """
    Generate the matrix (1,x) of size m x p+1
    :param x: Data matrix.
    :return: The matrix with 1 column of ones and the data matrix afterwards.
    """
    m, p = x.shape
    df = np.ones((m, p+1))
    df[:, 1:] = x
    return df


def preprocessing(df, objective, columns):
    """
    Normalize every column except for the class column. Called objective.
    :param df: Original source of data.
    :param objective: Name of the objective variable (class)
    :param columns: Names of all columns of interest.
    :return: The matrix with interest columns normalized and the objective column (class)
    """
    df = df[columns]
    m_class = df[objective]
    columns.remove(objective)
    df = df[columns].apply(lambda x: (x - x.mean()) / x.std())
    df[objective] = m_class
    return df


def genplot(y, x, w):
    """
    Method that generates the scatterplot of the points with the regression line.
    :param y: Vector of true values.
    :param x: Data.
    :param w: Coefficients of the linear model.
    :return: Creates the image. ID.PNG
    """
    fig, x_min, x_max, eps = plt.figure(), x.min(), x.max(), 1.0
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
                    alpha=2e-2,
                    itr=2e2,
                    eps=1e-9)
    print(w)
    print(lib.r2(y=y, c=x, x=w))

    genplot(y=y, x=np.array(df['AGST'].values), w=w)

    x = gendata(df.as_matrix(columns=['AGST', 'WinterRain', 'HarvestRain', 'Age']))

    w = lib.descent(y=np.array(df['Price'].values),
                    x=x,
                    alpha=2e-2,
                    itr=2e2,
                    eps=1e-9)
    print(w)
    print(lib.r2(y=y, c=x, x=w))

if __name__ == '__main__':
    main()
