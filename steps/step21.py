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
        # print('backward() in Exp: gx:', type(gx), 'gy:', type(gy))
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

# 역전파 활성화, 비활성화 모드
class Config:
    enable_backprop = True

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)   # as_array: x1이 float나 int인 경우 ndarray인스턴스로 변환, 그리고 Variable로 변환
    return Add()(x0, x1)

def mul(x0, x1):
    x0 = as_array(x0)
    x1 = as_array(x1)
    return Mul()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 인수로 주어진 객체를 Variable인스턴스로 변환
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

Variable.__mul__ = mul
Variable.__add__ = add
# 21.3 문제점 1
Variable.__rmul__ = mul
Variable.__radd__ = add


# 21.1
# x = Variable(np.array(2.0))
# y = x+np.array(3.0)
# print(y)
# variable(5.0)

# 21.2
# x = Variable(np.array(2.0))
# y = x+3.0
# print(y)
# variable(5.0)

# 21.3
# x = Variable(np.array(2.0))
# y = 3.0 * x + 1.0
# print(y)
# variable(7.0), mul도 as_array 처리해줘야됨

# 21.4
x = Variable(np.array([1.0]))
y = np.array([2.0])+x
print(y)