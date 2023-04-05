if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

# 28.2 로젠브록 함수 미분 
def rosenbrock(x0, x1):
    y = 100 * (x1- x0 **2) ** 2 + (1- x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)
# -2.0 400.0

# 28.3 경사하강법 구현
lr = 0.001
iters = 50000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

# plot_dot_graph(y, verbose = False, to_file='gradient.png')
# 그래프가 잘 그려지진 않음... 당연함 이건 network graph를 그리는거임

# ...
# variable(0.6832714656167057) variable(0.4653482760249264)
# variable(0.6834917840420289) variable(0.46565059996514135)