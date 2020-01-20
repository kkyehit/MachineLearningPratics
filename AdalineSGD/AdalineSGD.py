import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineSGD(object):
    """
    매개 변수
    ---------
    eta : float
        학습률 ( 0.0 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    shuffle : bool (defualt : true)
        True 라면 같은 반복이 되지 않도록 epoch마다 훈련데이터를 섞는다.
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성 시드

    속성
    ---------
    w_ :1d-array
        학습된 가중치
    errors_ : list
        에포크마다 누정된 분류 오류
    """
    def  __init__(self, eta = 0.01, n_iter = 50, shuffle = True, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False  
    
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
        self._initialized_weights(X.shape[1])   #"""랜덤한 작은 수로 가중치를 초기화 한다(w_를 초기화)"
        self.cost_ = []                         #"""각 시도마다 비용 저장"""
        for i in range(self.n_iter):        
            if self.shuffle:                    
                X, y = self._shuffle(X, y)      #"""특성X와 레이블 y의 순서를 섞는다."""
            cost = []
            for xi, target in zip(X, y):        
                cost.append(self._update_weights(xi, target))   #"""각 훈련 데이터의 가중치를 업데이트 하고 비용저장"""
            avg_cost = sum(cost)/len(y)                         #"""len(y) : y배열의 전체 요소를 반환""""
            self.cost_.append(avg_cost)
        return self

    "훈련 데이터 셋을 섞는다."
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))      #"""permutation(len(y)) :len(y)의 순서를 바꾼 배열을 반환한다."""
        print(r)
        return X[r], y[r]                       
        
    "가중치를 무작위의 작은 수로 초기화"
    def _initialized_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m)
        self.w_initialized = True
        
    "Adaline 학습 규칙을 적용하여 가중치 업데이트"
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error) 
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    "가중치마다 다시 초기화 하지 않아 스트리밍 데이터를 사용하는 온라인 학습방식을 모델을 훈련하기 위해 사용"
    "각 샘플마다 partial_fit 메서드 호출"
    "ada.partial_fit(X_std[0, :], y[0])"
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    "최종 입력 계산"
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
   
    "선형 활서화 계산, "
    def activation(self, X):
        return X

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


ada = AdalineSGD(eta = 0.01, n_iter=15)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()