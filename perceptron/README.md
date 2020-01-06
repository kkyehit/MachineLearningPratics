# Perceptron
***
- MCP 뉴런 모델을 기반
1. 가중치를 0 또는 무작위의 작은 값으로 초기화한다.
2. 각 훈련 샘플 X^(i)에서
    + 출력 값 계산
    + 가중치 업데이트
- 결정함수    
    + Φ(z) = 1 ( z > θ )
    +      = 0 ( other )
    + W0 = -θ, X0 = 1
    + W0 = -θ를 절편이라고 한다.
    + 입력 값 X과 이에 상응하는 가중치 벡터 W가 주어질 때 최종 입력(net input)은
        * z = W0 x X0 + ... + Wm x Xm = Σ(Xj x Wj) = W^T x X
- 가중치 벡터 w에 존재하는 개별 가중치 Wj에 대해 동시에 업데이트
    + Wj := Wj + △Wj
    + △Wj = η(y^(i) + ý^(i)) * Xj^(i)
        * η : 학습률, 만약 가중치가 0으로 초기화 되면 이 값은 벡터의 방향이 아니라 크기에만 영향을 미친다.
        * y^(i) : true class label 
        * ý^(i) : predicted class label
        * 모든 가중치 △Wj를 계산하기 전 ý^(i)를 다시 계산하지 않는다.
        * class label을 정확히 예측한 경우 가중치는 변경되지 않는다.
- Perceptron은 두 class가 선형적으로 구분되고 학습률이 충분히 작을 때만 수렴이 보장된다.
- 두 class를 선형적으로 구분할 수 없다면 훈련 데이터셋을 반복할 최대 횟수(epoch)를 지정하고 분류 허용 오차를 지정할 수 있다.
- OvA(One-versus-All) 전략을 사용하면 다중 클래스 분류로 확장할 수 있다.

***
## Artifical neuron
- 두개의 클래스가 있는 이진 분류 작업
    + 양성 클래스 : 임계값 보다 큰 경우
    + 음성 클래스 : 임계값 보다 작은 경우
- 입력 값 X와 이에 상응하는 가중치 벡터 W의 선형 조합으로 결정 함수 정의



***
## numpy ( Numberical Python )
- C언어로 구현된 파이썬 라이브러리로써 고성능의 수치 계산을 위해 사용.
- 백터 및 행렬 연산에 편리한 기능 제공

### numpy를 이용하여 array 정의
- data1 = [1, 2, 3, 4, 5]
- arr1 = np.array(data1)

### array의 크기 확인
- arr1.shape

### array의 자료형 확인
- arr1.dtype

### array 생성
- np.zero(n) 
    + n의 크기만큼 0으로 채운 array 생성 인자가 (3, 5)일 경우 0으로 채운 3행 5열 행렬 생성 
- np.ones(n) 
    + n의 크기만큼 1으로 채운 array 생성 인자가 (3, 5)일 경우 1으로 채운 3행 5열 행렬 생성 
- np.arange(a, b)
    + a부터 b - 1 까지 증가하는 값을 가진 array 생성
    + 만약 인자를 하나만 입력하면 0 ~ b - 1 까지 증가하는 값을 가진 array 생성
- np.full()
    + 배열에 사용자가 지정한 값을 넣는다.
- np.eye()
    + 대각선이 1이고 나머지는 0인 2차원 배열을 생성한다.

### array 연산
- 기본적으로 크기가 서로 동일한 array끼리 연산이 가능하지만
- numpy에서 크기가 다른 array 사이에 연산이 가능하도록 브로드캐스트 기능을 사용한다.
- arr1 * arr2 는 일반적인 행렬 곱이 아닌 요소별로 곱셈이 진행된다.
- vector와 matrix의 곱셈을 구하기 위해서는 dot() 함수를 사용한다.
    + np.dot(arr1, arr2)

### array Slice
- arr2 = arr1[0:2, 0:2]

### Numpy 함수
- np.random.randn( 5, 3) 
    + 무작위 값(실수)을 가진 5x3 array 반환

- np.abs(arr1)
    + 각 요소의 절댓값 반환.

- np.aqrt(arr1)
    + 각 요소의 제곱근 반환.

- np.sum(arr1)
    + 전제 요소의 합을 반환

- np.sum(arr1, axis = 0)
    + 행 간의 합을 반환

- np.std(arr1)
    + 전체 요소의 표준편차, 분산, 최소값, 최대값 반환(std, var, min, max)

- np.sort(arr1)
    + 전체 요소에 대해서 오름차순으로 정렬

- np.where()
    + 조건에 해당하는 index 값을 찾기
        * np.where( x >= 3)
        * x에서 3과 같거나 큰 값을 가지는 위치 반환
    + np.where(self.net_input >= 0.0, 1, -1)
        * np.where(조건, 조건에 맞을 때 값, 조건과 다를 때 값)
        * 반복문을 수행하는 것보다 빠르다.
- np.unique(x) 
    + 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환
- np.meshgrid()
    + x 값의 배열과 y 값의 배열로 2차원 그리드 포인트를 생성한다.
    + 결과는 그리드 포인트의 x 값만을 표시하는 행렬과 y 값만을 표시하는 행렬 두 개로 분리하여 출력한다

### numpy.random.RandomState()
-  랜덤넘버 생성기인 랜덤함수들을 포함하는 클래스.
-  RandomState는 다양한 확률분포측면에 수 많은 랜덤넘버 생성기들을 가지고 있다.
-  매개변수는 난수 생성기 시드이다.
-  numpy.random.RandomState.normal
    + 정규분포에서 랜덤 표본을 추출한다.
    + loc: float 또는 float의 array_like으로 분포의 평균이다.
    + scale: float 또는 float의 array_like으로 분포의 표준편차이다. 
    + size: int 또는 또는 int의 array_like으로 선택사항이다. 
    + 반환값은 ndarray또는 scalar 값이다. 정규 분포에서 표본을 추출한다.


***
## pandas
- R의 기능을 도입하여 데이터의 통계 기능을 제공하며 CSV 파일의 읽고 쓰기 기능도 지원합니다.