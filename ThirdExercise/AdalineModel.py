import random

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
        return 1 if self.b + (self.w1 * data[0] + self.w2 * data[1]) > 0 else -1

    def train(self):
        error = 1
        count = 0
        while count < self.epochs and error != 0:
            for i, x in enumerate(self.x_train):
                target = self.y_train[i]
                y_in = self.b + (self.w1 * x[0] + self.w2 * x[1])
                self.w1 += self.learning_ratio * (target - y_in) * x[0]
                self.w2 += self.learning_ratio * (target - y_in) * x[1]
                self.b += self.learning_ratio * (target - y_in)
        return {'error':error,'weight1':self.w1,'weight2':self.w2,'bias':self.b}