import numpy as np
import unittest
import weakref

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')


        self.data = data
        self.grad = None    # 미분값 저장
        self.creator = None # 함수와 변수의 관계
        self.generation = 0     # 세대 수를 기록하는 변수
    
    def set_creator(self, func):
        """creator 설정"""
        self.creator = func
        self.generation = func.generation + 1   # 세대를 기록한다(부모 세대 + 1)
    
    def backward(self):
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
    
    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):        # 별표를 붙임
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)          # 별표 붙여 언팩
        if not isinstance(ys, tuple):   # 튜플이 아닌 겨우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
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

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

for i in range(10):
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))