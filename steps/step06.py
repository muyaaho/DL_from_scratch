import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # 미분값 저장

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input  # 입력 변수를 기억(보관)한다.
        return output

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        print(f'Square forward x: {x}, return: {x**2}')
        return x**2
    
    
    def backward(self, gy):
        """x^2을 미분하는 함수

        Args:
            gy (Variable.grad): 이전 미분값

        Returns:
            Variable.grad: 역전파 계산된 값
        """
        
        x = self.input.data
        print(f'Square backward x: {x}, gy: {gy}')
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        print(f'Exp forward x: {x}, return: {np.exp(x)}')
        return np.exp(x)
    
    def backward(self, gy):
        """출력 쪽에서 전해지는 미분값 전달, e^x 미분해도 e^x이므로 그냥.. 계산

        Args:
            gy (Variable.grad): 이전 미분값

        Returns:
            Variable.grad: 역전파 계산된 값 
        """
        x = self.input.data
        print(f'Exp backward x: {x}, gy: {gy}')
        gx = np.exp(x) * gy
        # print('backward() in Exp: gx:', type(gx), 'gy:', type(gy))
        return gx

# 순전파하는 코드
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print()

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)