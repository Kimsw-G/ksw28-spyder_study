import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

print(perch_length.shape)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(perch_length, perch_weight, random_state=42)
# reshape를 통해 2차원 배열로 변경
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input, test_input)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor();
for n in range(1,40):
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    print(n, knr.score(test_input,test_target), knr.score(train_input,train_target))
    print()
    
knr.n_neighbors = 3
knr.fit(train_input, train_target)


# 어느정도의 오차가 생기는지 검사가능
# 결과값 19, 19그램 정도의 오차가 남을 알 수 이따
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target, test_prediction)
print(mae)

print(knr.predict([[25]]))


import matplotlib.pyplot as plt
plt.scatter(perch_length,perch_weight)
plt.scatter(25,knr.predict([[25]]))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()