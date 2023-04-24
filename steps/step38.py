if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)
'''
forward (2,3) -> (6,)
backward (6,) -> (2,3)
'''

x = Variable(np.random.randn(1,2,3))
y = x.reshape((2,3))
y = x.reshape(2,3)

x = Variable(np.random.randn(1,2,3))
y = F.transpose(x)
y.backward()
print(x.grad)