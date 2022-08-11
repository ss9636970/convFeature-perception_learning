import numpy as np

class Perception:
    def __init__(self, features_n=2048, cls_n=50):
        self.Weights = np.random.rand(features_n, cls_n)
        self.Bias = np.random.rand(1, cls_n)
        self.featuresN = features_n
        self.clsN = cls_n
        
        self.Weights_grad = None
        self.Bias_grad = None
        
        self.inputs = None
        self.outputs = None
        self.y = None
        
    def forward(self, inputs):
        n = inputs.shape[0]
        self.inputs = inputs
        self.outputs = np.matmul(self.inputs, self.Weights) + self.Bias
        self.y = self.softmax(self.outputs)
        return self.y
        
    def softmax(self, inputs):
        a, b = inputs.shape
        outputs = np.exp(inputs)
        sums = np.sum(outputs, axis=1)
        for i in range(a):
            outputs[i, :] = outputs[i, :] / sums[i]
        return outputs
    
    def weight_backward(self, inputs):
        n = inputs.shape[0]
        weight_grad = inputs.T
        bias_grad = np.ones([1, n])
        return weight_grad, bias_grad   # feature_n * N,  1 * clsN
    
    def softmax_backward(self, inputs):
        a, b = inputs.shape
        outputs = np.exp(inputs)
        sums = np.sum(outputs, axis=1)
        for i in range(a):
            outputs[i, :] = outputs[i, :] / sums[i]
        return outputs   # N * clsN
    
    def backward(self):
        wG, bG = self.weight_backward(self.inputs)
        softmaxG = self.softmax_backward(self.outputs)
        outputs = [wG, bG, softmaxG]
        return outputs

    def save_model(self):
        return {'W': self.Weights, 'B': self.Bias}

    def load_model(self, d):
        self.Weights = d['W']
        self.Bias = d['B']

class Lossfn:
    def __init__(self):
        self.real = None
    
    def forward(self, inputs, realClass):
        n = inputs.shape[0]
        self.real = realClass
        loss = 0.
        for i in range(n):
            loss += np.log(inputs[i, realClass[i]])
        loss = (-1) * loss / n
        return loss
    
    def backward(self, inputs):
        n, k = inputs.shape
        outputs = np.zeros([n, k])
        for i in range(n):
            c = (-1) / inputs[i, self.real[i]]
            outputs[i, self.real[i]] = c / n
        return outputs      # N * clsN

class Optimize:
    def __init__(self, model=None, lossf=None, lr=None):
        self.model = model
        self.lossf = lossf
        self.lr = lr
        
    def backward(self):
        modelG = self.model.backward()
        lossfG = self.lossf.backward(self.model.y)
        output1 = modelG[-1] * lossfG
        self.model.Weights_grad, self.model.Bias_grad = np.matmul(modelG[0], output1), np.matmul(modelG[1], output1)

    def upgrade_w(self):
        self.model.Weights = self.model.Weights - self.lr * self.model.Weights_grad
        self.model.Bias = self.model.Bias - self.lr * self.model.Bias_grad
        
    def zero_grad(self):
        self.model.Weights_grad = None
        self.model.Bias_grad = None