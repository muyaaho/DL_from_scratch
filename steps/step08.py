import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # 미분값 저장
        self.creator = None # 함수와 변수의 관계
    
    def set_creator(self, func):
        """creator 설정"""
        self.creator = func
    
    def backward(self):
        funcs = [self.creator]  
        while funcs:
            f = funcs.pop()     # 호출할 함수 꺼내기            
            x, y = f.input, f.output    # 함수의 입, 출력 가져오기
            x.grad = f.backward(y.grad) # backward 메서드 호출

            if x.creator is not None:   # x.creator가 있다면
                funcs.append(x.creator) # 함수를 funcs에 차례로 집어넣음, 이전 함수를 리스트에 추가함

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)    # 출력 변수에 창조자를 설정함
        self.input = input  # 입력 변수를 기억(보관)한다.
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

y.grad = np.array(1.0)
y.backward()
print(x.grad)