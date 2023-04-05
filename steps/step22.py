import numpy as np
import unittest
import weakref

class Variable:
    __array_priority__ = 200        # Variable의 연산자 우선순위 높이기
    def __init__(self, data, name = None):  # 변수에 이름 붙여주도록 설정
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')


        self.data = data
        self.name = name    # 계산그래프 시각화 할 때 변수 이름을 그래프에 표시할 수 있음
        self.grad = None    # 미분값 저장
        self.creator = None # 함수와 변수의 관계
        self.generation = 0     # 세대 수를 기록하는 변수
    
    def set_creator(self, func):
        """creator 설정"""
        self.creator = func
        self.generation = func.generation + 1   # 세대를 기록한다(부모 세대 + 1)
    
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()        # funcs 리스트에 같은 함수 중복추가 방지

        # 중첩메서드
        #   감싸는 메서드(backward) 안에서만 이용한다.
        #   감싸는 메서드(backward)에 정의된 변수(funcs, seen_set)을 사용해야 한다
        def add_func(f):            # DeZero 함수를 리스트에 추가
            if f not in seen_set:   # 함수 리스트를 세대 순으로 정렬
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    add_func(x.creator)     # 수정 전: funcs.append(x.creator)
        
        if not retain_grad:
            for y in f.outputs:
                y().grad = None # y는 약한 참조
    
    def cleargrad(self):
        self.grad = None
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n' + ' '*9)
        return 'variable('+p+')'

    def __mul__(self, other):
        return mul(self, other)
    
class Function:
    def __call__(self, *inputs):        # 별표를 붙임
        inputs = [as_variable(x) for x in inputs]       # as_variable 함수 이용하도록
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)          # 별표 붙여 언팩
        if not isinstance(ys, tuple):   # 튜플이 아닌 겨우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   # 세대 설정
            for output in outputs:
                output.set_creator(self)    # 연결 설정
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2
    
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy
    

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0*x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0

# 22.1 음수(부호 변환)
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

# 22.2 뺄셈
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

# 반대로 들어오는 경우 처리
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

# 22.3 나눗셈
class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0 / x1 **2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

# 22.4 거듭제곱
class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1)*gy
        return gx

def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow

# 역전파 활성화, 비활성화 모드
class Config:
    enable_backprop = True

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)   
    return Add()(x0, x1)

def mul(x0, x1):
    x0 = as_array(x0)
    x1 = as_array(x1)
    return Mul()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

Variable.__mul__ = mul
Variable.__add__ = add
Variable.__rmul__ = mul
Variable.__radd__ = add
Variable.__neg__ = neg      # 22.1
Variable.__sub__ = sub      # 22.2
Variable.__rsub__ = rsub    # 22.2

# 22.1
# x = Variable(np.array(2.0))
# y = -x
# print(y)
# variable(-2.0)

# x = Variable(np.array(2.0))
# y1 = 2.0 - x
# y2 = x - 1.0
# print(y1)
# print(y2)
# variable(0.0)
# variable(1.0)

# 22.4
x = Variable(np.array(2.0))
y = x ** 3
print(y)
# variable(8.0)