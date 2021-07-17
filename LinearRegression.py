import numpy as np
import matplotlib.pyplot as plt

def hypoFunction(x, theta):
    h = np.dot(x, theta)
    return h


def costFunction(h, y):
    """
    代价函数
    h:hypothesis,
    theta:特征向量系数
    y：特征值对应的实际值
    """
    m = len(y)
    J = 1 / (2 * m) * np.sum(np.power(h - y, 2))
    return J


def gradientDecent(x, y, h, theta, alpha, number):
    """梯度下降函数
    number：设置的梯度下降次数"""
    # for i in range(number):
    m = len(y)
    n = len(theta)
    J_history = np.zeros((number,1))
    for i in range(number):
        theta = theta - (alpha/m) * x.T.dot(h-y)
        h = hypoFunction(x, theta)
        J_history[i] = costFunction(h,y)
    print(theta)
    return h

def paint(x,y,hypothesis):
    plt.plot(x,y,"ro")
    plt.plot(x,hypothesis)
    plt.show()



def main():
    x = np.array([[1,1], [1,2], [1,3], [1,4], [1,5], [1,6]])
    y = np.array([[1], [2], [3], [4], [5], [6]])
    theta = np.array([[10],[0]])
    alpha = 0.1
    h = hypoFunction(x, theta)
    J = costFunction(h, y)
    h= gradientDecent(x, y, h, theta, alpha, 20000)
    x = x[:,-1]
    print(x)
    paint(x,y,h)
    pass


if __name__ == "__main__":
    main()