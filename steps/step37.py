if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import dezero.functions as F
from dezero import Variable

x = Variable(np.array(1.0))
y = F.sin(x)
print(y)

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.sin(x)
print(y)

x = Variable(np.array([[1,2,3], [4,5,6]]))
c = Variable(np.array([[10,20,30], [40,50,60]]))
t = x+c
y = F.sum(t)
print(y)