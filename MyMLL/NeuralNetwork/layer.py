import numpy as np


class Layer:
    def __init__(self, size, lastLayerSize, activationFunc='tanh', costFunction=None):

        self.weights = np.random.random((size, lastLayerSize))
        self.weightsDelta = np.random.random((size, lastLayerSize))
        self.error = np.zeros((size))
        self.input = np.zeros((lastLayerSize))
        self.values = np.zeros((size))
        self.valuesActivated = np.zeros((size))
        self.bias = np.ones((size))

        self.activationFunc = activationFunc
        self.costFunction = costFunction

    def feedForward(self, input):
        self.input = input
        self.values = np.dot(self.input, self.weights) + self.bias
        self.valuesActivated = self.activate(self.values)
        return self.valuesActivated

    def backProp(self, error=None, label=None):
        if label != None:
            self.error = self.cost(self.valuesActivated, label, True) * \
                self.activate(self.values, True)
        else:
            self.error = np.dot(
                self.weights.T, self.error)**self.activate(self.values, True)
        self.weightsDelta += np.outer(self.error.T, self.input)

        return self.error

    def update(self, batchSize, learningRate):
        self.weights = self.weightsDelta * learningRate / batchSize

    def activate(self, x, derv=False):
        if derv:
            if self.activationFunc == 'tanh':
                return 1 - np.tanh(x)**2
            elif self.activationFunc == 'sig' or self.activationFunc == 'sigmoid':
                return (np.tanh(x / 2) / 2 + 0.5) * (1 - np.tanh(x / 2) / 2 + 0.5)
            elif self.activationFunc == 'ReLU':
                x[x < 0] = 1 / 50
                x[x >= 0] = 1
                return x
        else:
            if self.activationFunc == 'tanh':
                return np.tanh(x)
            elif self.activationFunc == 'sig' or self.activationFunc == 'sigmoid':
                return np.tanh(x / 2) / 2 + 0.5
            elif self.activationFunc.lower() == 'relu':
                x[x < 0] = x / 50
                return x

    def cost(self, x, label, derv=False):
        if derv:
            if self.costFunction.lower() == 'mse' or self.costFunction.lower() == 'mean squared error' or self.costFunction.lower() == 'meansquarederror':
                return (x - label)
        else:
             if self.costFunction.lower() == 'mse' or self.costFunction.lower() == 'mean squared error' or self.costFunction.lower() == 'meansquarederror':
                return 1/2*(x - label)**2
