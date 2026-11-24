from sklearn.tree import DecisionTreeClassifier

# Features: [hours_studied, hours_slept]; label: 1 = pass, 0 = fail
X = [[1, 4], [2, 5], [3, 6], [5, 7], [6, 6], [8, 8]]
y = [0, 0, 0, 1, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)                  # training: learn rules from data
print(model.predict([[4, 7]]))   # inference: apply them to new data
