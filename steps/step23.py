if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x+3) **2 
y.backward()

# Variable backward 152p
# 18.2 부분을 넣지 않아서 에러났었음
print(y)
print(x.grad)
# variable(16.0)
# 8.0