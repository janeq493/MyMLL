import numpy as np
from random import randint
class ANN:
    def __init__(self,layer_info):
        np.random.seed(9130)

        self.num_layers = len(layer_info)
        
        self.weights = [np.random.random((layer_info[i],layer_info[i-1]))*2 -1 for i in range(1,self.num_layers)]
        print(self.weights)
        self.biases = [np.zeros((layer_info[i]))  +0.2for i in range(1,self.num_layers)]
        self.layers = [np.zeros((l)) for l in layer_info]
        self.layers_activated= [np.zeros((l)) for l in layer_info]
        self.h=[]
        self.temp = 0

    def feed_forward(self,data_vector):
        self.layers[0] = self.layers_activated[0] = data_vector
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            self.layers[i+1] = np.dot(w,self.layers_activated[i]) +b
            self.layers_activated[i+1]=self.activate(self.layers[i+1])
        return self.layers_activated[-1]

    def train(self,data,labels,learning_rate,batch_size,num_itterations):
        self.batch_size=batch_size
        self.delta_w = [np.zeros((w.shape)) for w in self.weights]
        self.delta_b = [np.zeros((b.shape)) for b in self.biases]
        j=0
        for i in range(num_itterations):
            i=randint(0,len(data)-1)
            j+=1
            self.feed_forward(data[i])
            self.back_prop(labels[i])
            if ((j)%self.batch_size==0):   
                j=0
                self.update(self.delta_w,self.delta_b,learning_rate)
                self.delta_w = [np.zeros((w.shape)) for w in self.weights]
                self.delta_b = [np.zeros((b.shape)) for b in self.biases]
                self.h.append(self.temp/self.batch_size)
                self.temp=0
        return self.h

    def back_prop(self,label):
        

        error = (self.layers_activated[-1]-label)*self.activate(self.layers[-1],True)
        self.delta_w[-1] += np.outer(error.T,self.layers_activated[-2])/self.batch_size
        self.delta_b[-1] += error/self.batch_size

        self.temp += (self.layers_activated[-1]-label)**2

        for i in range(2,self.num_layers):
            error = np.dot(self.weights[1-i].T,error)*self.activate(self.layers[-i],True)
            self.delta_w[-i]+=np.outer(error,self.layers_activated[-i-1])/self.batch_size
            self.delta_b[-i]+=error/self.batch_size
        

    def update(self,delta_w,delta_b,lr):
        for i in range(self.num_layers-1):
            self.weights[i]= self.weights[i]-(delta_w[i]*lr)
            self.biases[i]= self.biases[i]-(delta_b[i]*lr)
        #self.weights = [w-(dw*lr) for w,dw in zip(self.weights,delta_w)]
        #self.biases = [b-(db*lr) for b,db in zip(self.biases,delta_b)]

    def activate(self,x,derv=False):
        if derv:
            return 1-np.tanh(x)**2# np.where(x>0,1,1/100)
        return np.tanh(x) # np.where(x>0,x,x/100)