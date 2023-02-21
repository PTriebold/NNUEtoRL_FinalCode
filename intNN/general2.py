import numpy as np
from collections import deque
from numpy.typing import NDArray
from numpy import int8,int16,int32,int64,float64,float32
from typing import Tuple
from load_csv2 import load_csv
import pandas as pd

class inference:
    def __init__(self,dtypesmall:type = int16, dtypelarge:type = int64, upperLimit = 128) -> None: #x:NDArray ,y:NDArray, w:list, b:list, 
        self.dtypesmall = dtypesmall
        self.dtypelarge = dtypelarge
        
        self.dtypeLowerLimit, self.dtypeUpperLimit = self._getlimits(self.dtypesmall)
        self.upperLimit = upperLimit

        self.z = deque()
        self.a = deque()

    def _getlimits(self,dtype) -> tuple[int,int]:
        """Internal function to get the limits of the possible dtype's

        Args:
            dtype (np.dtype): dtype to analyse

        Returns:
            tuple(min, max): lower and upper limit of the specified dtype
        """
        try:
            return np.iinfo(dtype).min, np.iinfo(dtype).max
        except ValueError:
            return 1, 1

    def _getExtent(self, weights:list) -> int:
        """How large is the model

        Args:
            weights (list): list of weights

        Returns:
            int: how deep the model is
        """
        return len(weights)

    
    def relu(self, x:NDArray) -> NDArray:
        """Relu activation function 

        Args:
            x (np.ndarray): Layer on which the activation function is applied

        Returns:
            np.ndarray: Layer after the activation function
        """
        a = np.maximum(0,x)
        return a


    def softmax(self, x:NDArray) -> NDArray:
        """Softmax activation function

        Args:
            x (NDArray): Data to which the activation function is applied

        Returns:
            NDArray: Altered data
        """
        return self._fakesoftmax(x,self.dtypesmall,self.upperLimit)

    def _fakesoftmax(self, x:NDArray, dtype_out:type, dtypeoutmax:int) -> NDArray:
        """Fake softmax activation function. Due to the large numbers resulting from the exponential function of "large" integers, the normal softmax would just generate a overflow.

        Args:
            x (NDArray): input data
            dtype_out (type): output datatype
            dtypeoutmax (int): how is it going to be scaled?

        Returns:
            NDArray: Altered data
        """
        e = np.exp(x / (np.max(x,axis=1,keepdims=True) + np.finfo(float64).eps))
        return ((e / (np.sum(e, axis=1,keepdims=True))*(dtypeoutmax) + np.finfo(float64).eps)).astype(dtype_out)

    
    def forward_general(self, X:NDArray, weights:list, bias:list, activation:list, num_layers:int) -> None:
        """general forward calculation of an integer Neural Network

        Args:
            X (NDArray): Input Data
            weights (list): List of weights
            bias (list): List of biases
            activation (list): List of callable activation functions
            num_layers (int): Number of layers
        """
        # First Layer
        W = weights[0]
        b = bias[0]
        self.z.append(self.calcZ(X,W,b))
        self.a.append(activation[0](self.z[-1]))

        # Rest of them
        for i in range(1, num_layers):
            W = weights[i]
            b = bias[i]
            self.z.append(self.calcZ(self.a[-1],W,b))
            self.a.append(activation[i](self.z[-1]))

    def calcZ(self, x:NDArray, w:NDArray, b:NDArray) -> NDArray:  
        """Calculate the matrix multiplication and add t

        Args:
            x (NDArray): input data
            w (NDArray): weights
            b (NDArray): bias

        Returns:
            NDArray: calculated Z
        """
        z = np.matmul(x,w,dtype=np.float64)
        d = self.div(z,self.upperLimit)
        return self.dtypelarge(d) + b  

    def div(self, a:NDArray, b:NDArray) -> NDArray:
        """Division function. In tracing it is easier to track how much time a function took than just a math operation

        Args:
            a (NDArray): denominator
            b (NDArray): numerator

        Returns:
            NDArray: division result
        """
        return a / b

    def evaluate(self, Y:NDArray) -> None:
        """Evaluate the predicted data to get an accuracy score

        Args:
            Y (NDArray): Target data
        """
        self.predicted = np.argmax(self.a[-1], axis=1)
        self.target = np.argmax(Y,axis=1)
        self.result = self.predicted == self.target
        self.acc = np.count_nonzero(self.result) / Y.shape[0]
        self.resultDF = pd.DataFrame({
                                      "target" : self.target,
                                      "predicted" : self.predicted,
                                      "result" : self.result,
                                      "certainty" : np.round(np.max(self.a[-1]/self.dtypeUpperLimit,axis=1),2)
                                    })

def gesamt() -> float:
    """Helper function

    Returns:
        float: accuracy
    """
    inf = inference(upperLimit=128)
    loader = load_csv()
    W,b,x,y,activ,start,num = loader.load_general(1,"Kout",inf.relu,inf.relu,inf.softmax,inf.upperLimit,inf.dtypesmall)
    inf.forward_general(x,W,b,activ,num-start)
    inf.evaluate(y)
    print(inf.acc)
    return inf.acc

if __name__ == "__main__":
    gesamt()