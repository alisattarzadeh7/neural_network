import random

x_train = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
y_train = [-1, -1, -1, 1]
learning_ratio = 0.01
epochs = 100
w1 = random.uniform(-0.2, 0.2)
w2 = random.uniform(-0.2, 0.2)
b = random.uniform(-0.2, 0.2)
error = 1
count = 0
while count < epochs and error != 0:
    error = 0
    for i, x in enumerate(x_train):
        target = y_train[i]
        output = -1
        pred = b + (w1 * x[0] + w2 * x[1])
        if pred > 0:
            output = 1

        if output != target:
            error += 1
            w1 += learning_ratio * target * x[0]
            w2 += learning_ratio * target * x[1]
            b += learning_ratio * target

        # print("output " + str(output) + " target " + str(target))
        # print("ERROR " + str(error))
    count += 1
print("ENDING ERROR" + str(error))
print(" w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))
