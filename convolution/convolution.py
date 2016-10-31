import numpy as np
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap


@timing
def matrixMultiplication(imagei, filterf):
    """ convolution using matrix multiplication
        require n1 >= m1 and n2 >= m2
        without flip
    :param imagei: numpy array n1 x n2
    :param filterf: numpy array m1 x m2
    :return:
    """
    n1, n2 = imagei.shape
    m1, m2 = filterf.shape

    p = n1 - m1 + 1  #column num
    q = n2 - m2 + 1  #row num
    output = np.zeros((p, q))

    for i in range(p):
        for j in range(q):
            img = imagei[i:i + m1, j:j + m2]
            out = np.multiply(img, filterf)
            output[i][j] = np.sum(out)

    return output




def generateMatrix(m, n):
    """

    :param m: scala: row number
    :param n: scala: column number
    :return:
    """
    mat = np.random.randn(m, n)

    return mat


def main():
    """"""

    '''Question 3.1'''
    I1 = generateMatrix(128, 64)
    f1 = generateMatrix(64, 64)

    matrixMultiplication(I1, f1)

if __name__ == "__main__":
    main()
