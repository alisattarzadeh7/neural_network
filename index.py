from PerceptronModel import Model
x_train = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
y_train = [-1, 1, 1, 1]
learning_ratio = 0.01
epochs = 100

model = Model(x_train,y_train,learning_ratio,epochs)
model.train()
result = model.predict([-1,-1])
print(result)