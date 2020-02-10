#사이킷 런으로 구현한 perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np 

#사이킷 런의 datasets에서 iris(붓꽃)데이터 가져오기
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('클래스 레이블: ',np.unique(y))

#train_test_split를 이용하여 테스트 데이터와 훈련 데이터 나누기
#test_size = 테스트 데이터의 비율, 0.3이므로 30%
#random_state = 무작위로 섞기위한 시드
#stratify = 계층화 기능, 훈련 세트와 테스트 세트의 클레스 레이블을 동일하게 만든다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)
print("y의 클래스 레이블 : ", np.bincount(y))
print("y_train의 클래스 레이블 : ", np.bincount(y_train))
print("y_test의 클래스 레이블 : ", np.bincount(y_test))

#특성을 표준화(특성 스케일 조정)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#perceptron
#max_iter = 반복할 횟수(epoch)
#eta0 = 학습률
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
ppn.fit(X_train_std, y_train)

#테스트 세트로 예측하기
y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수 : %d ' %(y_test != y_pred).sum())

print('정확도 : %.2f' %accuracy_score(y_test,y_pred))

def plot_decision_regions(X, y, classfier, test_idx=None, resolution = 0.02):
    #마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #결정 경계그리기
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
   
    #테스트 샘플 부각
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolors='black', alpha=1.0, linewidths=1, marker='o', s=100, label='test set')
        


X_combined_std = np.vstack((X_train_std, X_test_std))   #vstack:열의 수가 같은 두 개 이상의 배열을 위아래로 연결하여 행의 수가 더 많은 배열을 만든다
y_combined = np.hstack((y_train, y_test))               #hstack:행의 수가 같은 두 개 이상의 배열을 옆으로 연결하여 열의 수가 더 많은 배열을 만든다
                                                        #dstack:깊이 방향으로 배열을 합치는 것으로 가장 안쪽에 존재하는 원소의 차원이 증가한다.
plot_decision_regions(X = X_combined_std, y = y_combined, classfier = ppn, test_idx = range(105, 150))
plt.xlabel('petal length [standardized]')
plt.xlabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
