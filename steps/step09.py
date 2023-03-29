import numpy as np

class Variable:
    def __init__(self, data):
        # 9.3 ndarray아니면 에러
        if data is not None:    # 데이터가 None이 아닐 때
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

        self.data = data
        self.grad = None    # 미분값 저장
        self.creator = None # 함수와 변수의 관계
    
    def set_creator(self, func):
        """creator 설정"""
        self.creator = func
    
    def backward(self):
        # 9.2
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            # np.ones_like: Variable의 data와 grad 데이터 타입을 같게 하기 위해

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))  # 9.3 결과값에 ndarray 적용
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

# 9.1 square, exp 파이썬 함수로 구현
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# 9.3 스칼라 타입이면 ndarray로 바꾸기
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(0.5))
# a = square(x)             # 9.1 함수 사용
# b = exp(a)
# y = square(b)
y = square(exp(square(x)))  # 9.1 연속해서 사용

# y.grad = np.array(1.0)    # 9.2
y.backward()                # backward() 호출하는 것만으로 가능!
print(x.grad)