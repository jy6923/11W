# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 해당 부분은 직접 작성 바랍니다 '''
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
# 와인 데이터 로드
wine = load_wine()

# DataFrame으로 변환
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# feature와 target 분리
feature = df.drop(columns='target')
target = df['target']

# train/test 분할
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

param_grid = {
     'criterion': ['gini', 'entropy'],
     'max_depth': [2, 3, 4, 5],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4]
}


clf = DecisionTreeClassifier()

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross validation
    scoring='accuracy',  # 평가 지표
    verbose=1,
    n_jobs=-1  # 모든 코어 사용
)

grid_search.fit(X_train, y_train)
# 예측
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# 테스트 세트 정확도 확인
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 해당 부분은 직접 작성 바랍니다 '''
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


# train/test 분할
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

param_grid = {
     'max_depth': [3, 5, 7, 9, 15], 
     'learning_rate': [0.1, 0.01, 0.001],
     'n_estimators': [50, 100, 200, 300]
}


xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,  # 5-fold cross validation
    scoring='accuracy',  # 평가 지표
    verbose=1,
    n_jobs=-1  # 모든 코어 사용
)

grid_search.fit(X_train, y_train)
# 예측
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# 테스트 세트 정확도 확인
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
