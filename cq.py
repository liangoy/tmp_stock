import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_bp = pd.read_csv('data/bp.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # gspc
data_hs = pd.read_csv('data/hs.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # hsi
data_jp = pd.read_csv('data/jp.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # n225
data_ax = pd.read_csv('data/ax.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # axjo
data_uk = pd.read_csv('data/uk.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # FTSE
data_sh50 = pd.read_csv('data/sh50.csv').dropna().drop(['Adj Close', 'Volume'], axis=1)  # sh000001


def return_rate_list(stock_data=None, long=1):
    '''

    :param stock_data: 股票价格的列表
    :param long: 买卖间隔时长,单位是天
    :return: 返回股票收益率的列表
    '''
    data = np.array(stock_data)
    data_start = data[:-long]
    data_end = data[long:]
    return_rate = data_end / data_start - 1
    return return_rate


def return_rate_by_time(stock_data, dmax=1000):
    '''

    :param stock_data: 股票价格的列表
    :param dmax: 最大买卖间隔时长,单位是天
    :return: 返回一个股票收益偏度的列表,列表的第0项是买卖当间隔天数为1时的偏度,列表的第2项是买卖当间隔天数为2时的偏度......
    '''
    lis = []
    for i in range(1, dmax):
        rate = return_rate_list(stock_data, long=i)
        skew = pd.Series(rate).skew()
        lis.append(skew)
    return lis


def frequency(stock_data=None, number=None):
    '''

    :param stock_data: 股票价格的列表
    :param number: 要查看的数字
    :return: 返回数字的频率序列
    '''
    data = np.array(stock_data, dtype=np.int)
    data = data % 10
    return data == number


def frequency_number_rank(stock_data):
    '''
    求股票价格整数部分个位数的频率
    :param stock_data: 股票价格的列表
    :return: 返回一个列表,元素由数字及其对应的频率构成,列表按照频率由小到大排列
    '''
    data = [sum(frequency(stock_data, i)) / len(stock_data) for i in range(10)]
    return sorted(enumerate(data), key=lambda x: x[1])


def z_test(lis, mean, c=1.96):
    '''
    置信检验
    :param lis:列表
    :param mean: 真实分布的均值
    :param c: 标准正泰分布中置信水平对应的值,例如0.99对应的c为2.58,0.95对应1.96,0.90对应1.64等等
    :return:返回布尔值,True代表通过,False代表不通过
    '''
    n = np.array(lis)
    t = c * n.std() / len(n) ** 0.05
    return n.mean() - t < mean < n.mean() + t


if __name__ == '__main__':
    data_list = [data_bp, data_hs, data_jp, data_ax, data_uk, data_sh50]
    '''
    第二题
    '''
    colors = ['yellow', 'blue', 'black', 'green', 'purple', 'red']
    for i, c in zip(data_list, colors):
        plt.plot(return_rate_by_time(i.Close, dmax=1600), color=c)

    name = ['gspc', 'hsi', 'n225', 'axjo', 'ftse', 'sh000001']
    text = '\n'.join([c + ' : ' + n for n, c in zip(name, colors)])
    plt.text(x=0, y=2, s=text)
    plt.show()

    '''
    第三题
    '''
    low, high = [], []
    for i in data_list:
        output = frequency_number_rank(i.Close)
        # print(output)
        low.append(output[0][0])
        high.append(output[-1][0])
    n = np.array([[i, j] for i, j in zip(low, high)])
    print(n)
    print('不难看出,综合而言,出现次数最少的数字是4,并列最多的是0和5')

    # 以下为显著性检验

    dic = {True: 0, False: 0}
    for i in data_list:
        for j in range(10):
            f = frequency(i.Close, j)
            output = z_test(f, 0.1, c=1.64)
            dic[output] += 1
    print('0.90的置信水平下,检验结果为:', dic, '通过率为1.0,符合要求')
    print('所以,对于股指而言,尾数基本可以看成随机的,不必去研究其规律')
