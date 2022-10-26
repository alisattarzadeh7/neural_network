import random
#
# x_train = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
# y_train = [-1, -1, -1, 1]
# learning_ratio = 0.01
# epochs = 100
# w1 = random.uniform(-0.2, 0.2)
# w2 = random.uniform(-0.2, 0.2)
# b = random.uniform(-0.2, 0.2)
# error = 1
# count = 0
# while count < epochs and error != 0:
#     error = 0
#     for i, x in enumerate(x_train):
#         target = y_train[i]
#         output = -1
#         pred = b + (w1 * x[0] + w2 * x[1])
#         if pred > 0:
#             output = 1
#
#         if output != target:
#             error += 1
#             w1 += learning_ratio * target * x[0]
#             w2 += learning_ratio * target * x[1]
#             b += learning_ratio * target
#
#         # print("output " + str(output) + " target " + str(target))
#         # print("ERROR " + str(error))
#     count += 1
# print("ENDING ERROR" + str(error))
# print(" w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))
#


class Model:

    w1 = random.uniform(-0.2, 0.2)
    w2 = random.uniform(-0.2, 0.2)
    b = random.uniform(-0.2, 0.2)

    def __init__(self, x_train, y_train, learning_ratio,epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.learning_ratio = learning_ratio
        self.epochs = epochs


    def predict(self,data):
        return 1 if self.b + (self.w1 * data[0] + self.w2 * data[1]) > 0  else -1

    def train(self):
        error = 1
        count = 0
        while count < self.epochs and error != 0:
            error = 0
            for i, x in enumerate(self.x_train):
                target = self.y_train[i]
                output = -1
                pred = self.b + (self.w1 * x[0] + self.w2 * x[1])
                if pred > 0:
                    output = 1

                if output != target:
                    error += 1
                    self.w1 += self.learning_ratio * target * x[0]
                    self.w2 += self.learning_ratio * target * x[1]
                    self.b += self.learning_ratio * target
            count += 1
        return {'error':error,'weight1':self.w1,'weight2':self.w2,'bias':self.b}