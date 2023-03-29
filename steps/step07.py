import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    
        self.creator = None # 함수와 변수의 관계 추가
    
    def set_creator(self, func):
        """creator 설정"""
        self.creator = func
    
    def backward(self):
        f = self.creator                    # 1. 함수 가져오기
        if f is not None:                   
            x = f.input                     # 2. 함수의 입력 가져오기
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드 호출
            x.backward()                    # 하나 앞 변수의 backward 메서드를 호출한다(재귀)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)    # 출력 변수에 창조자를 설정함
        self.input = input  
        self.output = output        # 출력도 저장
        return output

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2
    
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 순전파하는 코드
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라간다
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


y.grad = np.array(1.0)

# C = y.creator       # 1. 함수를 가져온다
# b = C.input         # 2. 함수의 입력을 가져온다
# b.grad = C.backward(y.grad) # 3. 함수의 backward 메서드를 호출한다.

# B = b.creator       # 1. 함수를 가져온다
# a = B.input         # 2. 함수의 입력을 가져온다
# a.grad = B.backward(b.grad)   # 3. 함수의 backward 메서드를 호출한다.

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

y.backward()
print(x.grad)