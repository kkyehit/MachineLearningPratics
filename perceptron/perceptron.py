import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    매개 변수
    ---------
    eta : float
        학습률 ( 0.0 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성 시드

    속성
    ---------
    w_ :1d-array
        학습된 가중치
    errors_ : list
        에포크마다 누정된 분류 오류
    """
    def  __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    
    "훈련 데이터 학습"
    def fit(self, X, y):
        """
        매개변수
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            n_sample개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_samples]
            타켓 값
        
        반환값
        ----------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)                         #"랜덤 시드 값 설정"
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])   #"정규화를 이용한 가중치 초기화"
        self.errors_ = []                                                       #"각 시도마다 에러 수 저장"

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):    #"zip() : 자료형을 묶어준다. (X[0] y[0]) ... (X[n] y[n])"
                update = self.eta * (target - self.predict(xi)) #"학습률 * ( 실제 값 - 예측 값 )"
                self.w_[1:] += update * xi  #"배열의 두번째 요소 부터 마지막까지 업데이트(개별 가중치에 대해 동시 업데이트)"
                self.w_[0] += update        #"배열의 첫번쨰 요소 업데이트(절편)"
                errors += int(update != 0.0)#"에러가 있다면 에러 수 증가"
            self.errors_.append(errors)     #"에러 배열에 에러 수 추가"
        return self
    
    "최종 입력 계산"
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] #"가중치와 훈련 샘플을 행렬 곱한 후 절편을 더한 값을 리턴한다."

    "단위 계단 함수를 사용하여 클레스 레이블을 반환"
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1) #"최종 입력 계산 결과가 0.0 보다 크면 1, 아니면 -1 리턴"


"pandas 라이브러리를 사용하여 UCI 머신 러닝 저장소에서 붓꽃 데이터 셋을 로드"
"df = pd.read_csv('https://archive.ics.uci.edu/ml' 'machine-learning-databases/iris/iris.data', header = None)"
df = pd.read_csv('../iris.data', header = None)

"100개의 훈련 샘플에서 첫 번째 열과 세 번째 열을 추출하여 특성 행렬 X에 저장"
y = df.iloc[0:100, 4].values            #"target 값 추출"
y = np.where(y == 'Iris-setosa', -1, 1) #"-1과 1로 표현"

X = df.iloc[0:100, [0, 2]].values       #"첫번째 열과 세번째 열을 가져온다."

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

"Perceptron 알고리즘 훈련"
"epoch 대비 잘못 분류된 오차를 그래프로 그린다"
ppn = Perceptron(eta = 0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epoch')
plt.ylabel('Number of errors')
plt.show()

"데이터셋의 결정 경계 시각화"
def plot_decision_regions(S, y, classfier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolors='black')

plot_decision_regions(X,y, classfier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()