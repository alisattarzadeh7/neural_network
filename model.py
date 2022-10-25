import random

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]
learning_ratio = 0.3
epoch = 100
w1 = random.uniform(-0.2, 0.2)
w2 = random.uniform(-0.2, 0.2)
b = random.uniform(-0.2, 0.2)
error = 1
count = 0
while count < epoch and error != 0:
    error = 0
    for i, array in enumerate(x_train):
        target = y_train[i]
        output = 0
        pred = w1 * array[0] + w2 * array[1] - b
        if (pred > 0):
            output = 1

        if (output != target):
            error += 1

        w1 += learning_ratio * (target - output) * array[0]
        w2 += learning_ratio * (target - output) * array[1]
        b += learning_ratio * target * (-1)
    count += 1
print("ENDING ERROR" + str(error))
print(" w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))
