import numpy as np

class Perception:
    def __init__(self, features_n=2048, cls_n=50):
        self.Weights = np.random.rand(features_n, 1024)
        self.Bias = np.random.rand(1, 1024)
        self.Weights2 = np.random.rand(1024, cls_n)
        self.Bias2 = np.random.rand(1, cls_n)
        self.featuresN = features_n
        self.clsN = cls_n
        
        self.Weights_grad = None
        self.Bias_grad = None
        self.Weights2_grad = None
        self.Bias2_grad = None
        
        self.inputs = None
        self.h1 = None
        self.outputs = None
        self.y = None
        
    def forward(self, inputs):
        n = inputs.shape[0]
        self.inputs = inputs
        self.h1 = (np.matmul(self.inputs, self.Weights) + self.Bias) / self.inputs.shape[1]
        self.outputs = (np.matmul(self.h1, self.Weights2) + self.Bias2) / self.h1.shape[1]
        self.y = self.softmax(self.outputs)
        return self.y
        
    def softmax(self, inputs):
        a, b = inputs.shape
        outputs = np.exp(inputs)
        sums = np.sum(outputs, axis=1).reshape([1, -1])
        outputs = outputs / sums.T
        return outputs
    
    def weight_backward(self, inputs):
        n = inputs.shape[0]
        weight_grad = inputs.T
        bias_grad = np.ones([1, n])
        return weight_grad, bias_grad   # feature_n * N,  1 * clsN

    def weight2_backward(self, inputs):
        n = inputs.shape[0]
        weight2_grad = inputs.T
        bias2_grad = np.ones([1, n])
        weightProcess_grad = self.Weights2.T
        return weight2_grad, bias2_grad, weightProcess_grad   # feature_n * N,  1 * clsN
    
    def softmax_backward(self, inputs):
        a, b = inputs.shape
        expInput = np.exp(inputs)
        expSums = np.sum(expInput, axis=1).reshape([1, -1])
        expSums2 = expSums ** 2
        term1 = expInput / expSums.T
        term2 = (-1) * expInput / expSums2.T
        return term1 + term2
    
    def backward(self):
        wG, bG = self.weight_backward(self.inputs)
        wG2, bG2, process = self.weight2_backward(self.h1)
        softmaxG = self.softmax_backward(self.outputs)
        outputs = [wG, bG, wG2, bG2, process, softmaxG]
        return outputs

    def save_model(self):
        return {'W': self.Weights, 'B': self.Bias, 'W2': self.Weights2, 'B2': self.Bias2}

    def load_model(self, d):
        self.Weights = d['W']
        self.Bias = d['B']
        self.Weights2 = d['W2']
        self.Bias2 = d['B2']

class Lossfn:
    def __init__(self):
        self.inputs = None
        self.real = None
    
    def forward(self, inputs, realClass):
        n = inputs.shape[0]
        self.inputs = inputs
        self.real = realClass
        loss = 0.
        for i in range(n):
            loss += np.log(self.inputs[i, realClass[i]])
        loss = (-1) * loss / n
        return loss
    
    def backward(self):
        n, k = self.inputs.shape
        outputs = np.zeros([n, k], dtype=np.float)
        for i in range(n):
            c = (-1) / self.inputs[i, self.real[i]]
            outputs[i, self.real[i]] = c / n
        return outputs      # N * clsN

class Optimize:
    def __init__(self, model=None, lossf=None, lr=None):
        self.model = model
        self.lossf = lossf
        self.lr = lr
        
    def backward(self):
        modelG = self.model.backward()
        lossfG = self.lossf.backward()
        output1 = modelG[-1] * lossfG
        self.model.Weights2_grad, self.model.Bias2_grad = np.matmul(modelG[2], output1), np.matmul(modelG[3], output1)
        output2 = np.matmul(output1, modelG[4])
        self.model.Weights_grad, self.model.Bias_grad = np.matmul(modelG[0], output2), np.matmul(modelG[1], output2)

    def upgrade_w(self):
        self.model.Weights = self.model.Weights - self.lr * self.model.Weights_grad
        self.model.Bias = self.model.Bias - self.lr * self.model.Bias_grad
        self.model.Weights2 = self.model.Weights2 - self.lr * self.model.Weights2_grad
        self.model.Bias2 = self.model.Bias2 - self.lr * self.model.Bias2_grad
        
    def zero_grad(self):
        self.model.Weights_grad = None
        self.model.Bias_grad = None
        self.model.Weights2_grad = None
        self.model.Bias2_grad = None