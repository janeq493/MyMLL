import numpy as np

class SVM:
    def __init__(self,num_of_weights,learning_rate):
        
        self.num_of_weights = num_of_weights
        self.learning_rate = learning_rate
        self.weights = np.ones((num_of_weights))

    def train(self,data_labels,data_values,num_of_itterations):

        self.data_set_length = len(data_values) 

        for i in range(1,num_of_itterations):
            predicted_output = np.dot(data_values[i],self.weights)
            print()
            if ((predicted_output*data_labels[i])<1):
                self.weights = self.weights + self.learning_rate*((data_values[i]*data_labels[i])+((-2/i)*self.weights))
            else:
                self.weights = self.weights + self.learning_rate*((-2/i)*self.weights)

    def get_prediciton(self,examples):
        results = []
        for i in range(1,len(examples)):
            predicted  = np.dot(examples[i],self.weights)
            if predicted > 0:
                predicted = 1
            else:
                predicted = 0
            results.append(predicted)
        return results





