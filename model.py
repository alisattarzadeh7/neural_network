import random

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0,0,0,1]
learning_ratio = 0.0001
epoch = 100000
w1 = 0
w2 = 0
b = 0
count = 0
while count < epoch :
    error = 0
    for array in x_train:
        target = array[2]
        output = 0
        prediction = w1 * array[0] + w2 * array[1] - b
        if (prediction > 0):
            output = 1
        else:
            output = 0

        if (output != target):
            error += 1
            w1 += learning_ratio * target * array[0]
            w2 += learning_ratio * target * array[1]
            b += target * learning_ratio

        print("output " + str(output) + " target " + str(target))
        print("ERROR " + str(error))
    count += 1
print("COUNT " + str(count))
print("ENDING ERROR" + str(error))
print(" w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))