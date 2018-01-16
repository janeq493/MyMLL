import numpy as np
from layer import Layer


class neuralNetwork:
    def __init__(self, inputSize):
        self.layerInfo = [inputSize]
        self.Layers = []

    def addLayer(self, size, activationFunc='tanh'):
        self.Layers.append(Layer(size, self.layerInfo[-1], activationFunc))
        self.layerInfo.append(size)

    def train(self, training, labels, learningRate, batchSize):
        for t, l in zip(training, labels):
            self.feedForward(t)
            self.backProp(l)
            self.update(batchSize, learningRate)

    def feedForward(self, input):
        for l in self.Layers:
            input = l.feedForward(input)
        return input

    def backProp(self, label):
        error = self.Layers[-1].backProp(label=label)
        for i in range(2, len(self.Layers)):
            error = self.Layers[-i].backProp(error)

    def update(self, batchSize, learningRate):
        for l in self.Layers:
            l.update(batchSize, learningRate)
