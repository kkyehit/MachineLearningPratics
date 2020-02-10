# scikit Learn 

    pip install scikit-learn 또는 python -m pip install scikit-learn 명령어를 통해 사이킷 런 설치

- 사이킷 런 라이브러리는 많은 학습 알고리즘과 데이터 전처리, 평가를 위해 사용할 수 있는 함수가 있다.
- 사이킷 런의 많은 함수와 매서드는 문자열 형태의 클레스 레이블을 다룰 수 있다.
- 사이킷 런의 model_selection 모듈의 train_test_split 함수를 사용하여 특성과 레이블을 테스트 데이터와 훈령 데이터로 랜덤하게 나눈다.
- train_test_split 함수는 계층화도 지원한다.
    + 계층화 : 훈련 세트와 테스트 세트의 클래스 레이블 비율을 입력 데이터셋과 동일하게 한다는 것을 의미한다.
- 사이킷 런의 preprocessing 모듈의 StandardScaler 클래스를 사용하여 특성을 표준화한다.(특성 스케일 조정)
    + sc = StandardScaler() : sc에 StandardScaler객체를 할당
    + sc.fit(X) = 각 특성 차원마다 샘플 평균과 표준 편차를 계산한다.
    + sc.transform(X) = 계산된 샘플 평균과 표준 편차를 이용하여 훈련 세트를 표준화 한다.
- 사이킷 런의 알고리즘은 대부분 OvR(One-versus-Rest)방식을 사용하여 다중 분류를 지원한다.
    + perceptron으로 다중 분류가 가능하다.
    + 그러나 perceptron으로는 선형적으로 구분되지 않는 데이터 셋에는 수렴하지 못한다.
- 사이킷 런 라이브러리는 mertics 모듈 아래에 다양한 성능 지표가 있다.