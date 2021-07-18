import numpy as np

def NormalEquation(x,y):
    """
    正态方程：默认假设函数为：h = theta0+theta1x+theta2x
    x:x矩阵，第一列设置为x0 = 1
    y:y矩阵
    return:返回theta矩阵
    """ 
    theta = (x.T.dot(x)).I.dot(x.T).dot(y)
    return theta.astype(dtype = int)

def main():
    x = np.mat([[1,1],[1,2]])
    y = np.mat([[3],[5]])
    theta = NormalEquation(x,y)
    print(theta)

if __name__ == "__main__":
    main()