import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []

        "최종입력 -> 활성화 -> 출력 순으로 진행"
        for i in range(self.n_iter):
            net_input = self.net_input(X)               "최종 입력 계산"
            output = self.activation(net_input)         "선형 활성화 계산"
            errors = ( y - output )                     "오차 계산 (실제 값 - 예측 값)"
            self.w_[1:] += self.eta * X.T.dot(errors)   "배열의 T 속성 : 2차원 배열의 전치(transpose) 연산이며 행과 열을 바꾸는 작업이다."
                                                        "가중치 변화량 : 학습률 * 비용 함수의 기울기"
                                                        "전체 X에 대하여 기울기를 계산한다."
            self.w_[0] += self.eta * errors.sum()       "절편에 errors수를 더한다."
            cost = (errors ** 2).sum() / 2.0            "비용 함수 =  ½Σ(실제 값 - 계산한 값)^2"
            self.cost_.append(cost)                     "현제 시도에서의 비용 함수 값 저장"
        return self
    
    "최종 입력 계산"
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
   
    "선형 활성화 계산, "
    def activation(self, X):
        return X    "선형 활성화 단계를 보여주기 위함"

    "단위 계단 함수를 사용하여 클레스 레이블을 반환"
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


"pandas 라이브러리를 사용하여 UCI 머신 러닝 저장소에서 붓꽃 데이터 셋을 로드"
"df = pd.read_csv('https://archive.ics.uci.edu/ml' 'machine-learning-databases/iris/iris.data', header = None)"
df = pd.read_csv('../iris.data', header = None)

"100개의 훈련 샘플에서 첫 번째 열과 세 번째 열을 추출하여 특성 행렬 X에 저장"
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

"Adaline 알고리즘 훈련"
"epoch 대비 잘못 분류된 오차를 그래프로 그린다"
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
ada1 = AdalineGD(eta = 0.01, n_iter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Leaning rate 0.01')

ada2 = AdalineGD(eta = 0.00001, n_iter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker = 'x')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Leaning rate 0.00001')
plt.show()
"학습률이 너무 크면 비용 전역 최소값을 지나치키 떄문에 함수를 최소화 하지 못하고 비용을 매 시도 마다 커진다."
"학습률이 너무 작으면 전역 최소값에 수렴하기 위해서는 많은 시도가 필요하다."



"데이터 표준화 ( 특성 스케일을 조정 하기 위해 )"
"경사 하강법은 특성 스케일을 조정하여 빠르게 전역 최소값으로 수렴하도록 할 수 있다."
"X_std : 표준화된 데이터"
X_std = np.copy(X)
X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

"데이터셋의 결정 경계 시각화"
def plot_decision_regions(S, y, classifier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = S[:, 0].min() - 1, S[:, 0].max() + 1
    x2_min, x2_max = S[:, 1].min() - 1, S[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = S[y == cl, 0], y = S[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolors='black')


ada3 = AdalineGD(eta = 0.01, n_iter=15)
ada3.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada3)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada3.cost_) + 1), np.log10(ada3.cost_), marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()