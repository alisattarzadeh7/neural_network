import random


class Model:
    w1 = random.uniform(-0.2, 0.2)
    w2 = random.uniform(-0.2, 0.2)
    b = random.uniform(-0.2, 0.2)
    errorThreshold = 0.002

    def __init__(self, x_train, y_train, learning_ratio, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.learning_ratio = learning_ratio
        self.epochs = epochs

    def predict(self, data):
        return 1 if self.b + (self.w1 * data[0] + self.w2 * data[1]) > 0 else -1

    def train(self):
        count = 0
        stopTraining = False
        while count < self.epochs and stopTraining == False:

            for i, x in enumerate(self.x_train):
                target = self.y_train[i]
                y_in = self.b + (self.w1 * x[0] + self.w2 * x[1])
                newWeight1 = self.learning_ratio * (target - y_in) * x[0]
                newWeight2 = self.learning_ratio * (target - y_in) * x[1]
                if abs(self.w1 - newWeight1) < self.errorThreshold or abs(self.w2 - newWeight2) < self.errorThreshold:
                    stopTraining = True
                    break
                self.w1 += newWeight1
                self.w2 += newWeight2
                self.b += self.learning_ratio * (target - y_in)

            count += 1
        print(f"model trained after {count} epochs")
        return {'weight1': self.w1, 'weight2': self.w2, 'bias': self.b}
