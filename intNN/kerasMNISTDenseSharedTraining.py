import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
from time import time_ns
from general2 import inference as inferer
import keras
from os.path import exists
from tqdm import tqdm

class kerasMNIST():
  def __init__(self, nDense:int) -> None:
    """Keras part of the comparison script. Trains and evaluates mnist data

    Args:
        nDense (int): How many hidden layers should be initialised
    """
    self.train_images = mnist.train_images()
    self.train_labels = mnist.train_labels()
    self.test_images = mnist.test_images()
    self.test_labels = mnist.test_labels()

    self.train_images = (self.train_images / 255) - 0.5
    self.test_images = (self.test_images / 255) - 0.5

    self.train_images = self.train_images.reshape((-1, 784))
    self.test_images = self.test_images.reshape((-1, 784))
    self.nDense = nDense

  def start(self, Breite:int = 64) -> None:
    """Start macro. Calls the build and train function for you

    Args:
        Breite (int, optional): How wide the hidden layers should be. Defaults to 64.
    """
    self.build_model(self.nDense,Breite)
    self.train()

  def saveModel(self, i:int) -> None:
    """Does what it says on the tin. It saves the model

    Args:
        i (int): number in the filename representing the number of hidden layers
    """
    self.model.save(f"Kout/model_d{i}.h5")

  def loadModel(self, i:int) -> None:
    """Again, does exactly what the name suggests. It loads a keras model.

    Args:
        i (int): Integer in the filename
    """
    self.model = keras.models.load_model(f"Kout/model_d{i}.h5")

  def build_model(self, nDense:int, Breite:int = 64) -> None:
    """Construct the Keras model with a specified amount of hidden layers and a specified width of those layers

    Args:
        nDense (int): How many dense layers should there be?
        Breite (int, optional): How wide should the dense layers be?. Defaults to 64.
    """
    self.model = Sequential()
    # Add the input layer
    self.model.add(Dense(Breite,activation="relu", input_shape=(784,)))

    # Add the specified amount of hidden layers
    for _ in range(nDense):
      self.model.add(Dense(Breite,activation="relu"))
  
    # Add an output layer and compile the whole model
    self.model.add(Dense(10, activation='softmax'))
    self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'],)

  def train(self) -> None:
    """Training the model. Will fail if self.model is not defined!
    """
    self.model.fit(self.train_images,to_categorical(self.train_labels),epochs=1,batch_size=32,verbose=0)


  def predict(self,GPU:bool = True, Appendix:str = "") -> int:
    """Inferes on the data, while saving the execution time

    Args:
        GPU (bool, optional): Run on GPU or CPU. Defaults to True.
        Appendix (str, optional): DEPRECATED! String that is appended to the end of the saved filename. Defaults to "".

    Returns:
        int: Time it took for the prediction
    """
    with tf.device("/GPU:0" if GPU else "/CPU:0"):
      tbe = time_ns()
      self.predictions = self.model.predict(self.test_images,verbose=0,batch_size=10000)
      taf = time_ns()

    return taf - tbe
    
  def eval(self) -> float:
    """Use the prediction results to determine how good the accuracy was.

    Returns:
        float: Accuracy score
    """
    self.pred = np.argmax(self.predictions,axis=1)
    self.score = np.count_nonzero(self.pred == self.test_labels) / self.test_labels.shape[0]

    return self.score

  def save(self) -> None:
    """Save the weights, biases, training and test data as well as the model to the harddrive.
    """
    for i, layer in enumerate(self.model.layers):
      weigths = layer.get_weights()
      pd.DataFrame(weigths[0]).to_csv(f"Kout/W{i+1}.txt",index=False,header=False)
      pd.DataFrame(weigths[1]).to_csv(f"Kout/b{i+1}.txt",index=False,header=False)
        
    if not exists("Kout/ytrain.txt"):
      pd.DataFrame(self.train_labels).to_csv("Kout/ytrain.txt",index=False,header=False)

    if not exists("Kout/xtrain.txt"):
      pd.DataFrame(self.train_images).to_csv("Kout/xtrain.txt",index=False,header=False)

    if not exists("Kout/ytest.txt"):
      pd.DataFrame(self.test_labels).to_csv("Kout/ytest.txt",index=False,header=False)

    if not exists("Kout/xtest.txt"):
      pd.DataFrame(self.test_images).to_csv("Kout/xtest.txt",index=False,header=False)

    with open("Kout/model.json","w") as file:
      file.write(self.model.to_json())
 
def loadModel(ker: kerasMNIST, userMax:int = 128) -> tuple[list,list,list,int,int]:
  """Load a model from keras to the intNN

  Args:
      ker (kerasMNIST): KerasMNIST class from which the stuff should be loadedd
      userMax (int, optional): Scaling factor for float 1. Defaults to 128.

  Returns:
      tuple[list,list,list,int,int]: Tuple of Weights, bias, activation, start int and number of layers
  """
  W = []
  b = []
  activation = []
  start = 0
  num = len(ker.model.layers)

  for i, layer in enumerate(ker.model.layers):
    weigths = layer.get_weights()
    W.append((weigths[0] * userMax).astype(np.int16))
    b.append((weigths[1] * userMax).astype(np.int16))
    activation.append(layer.activation.__name__)

  return W,b,activation,start,num
  
  

if __name__ == "__main__":
  x = mnist.test_images().reshape((-1,784)).astype(np.uint8)
  y = to_categorical(mnist.test_labels())


  df = pd.DataFrame()
  rf = pd.DataFrame()

  nlist = [64,128,256,512,1024,2048,4096]

  depth = 100
  width = len(nlist)

  for m in range(width):    
    neurons = nlist[m]

    df[f"GPU{neurons}"] = [0]*depth
    df[f"CPU{neurons}"] = [0]*depth
    df[f"INT{neurons}"] = [0]*depth

    rf[f"GPU{neurons}"] = [0]*depth
    rf[f"CPU{neurons}"] = [0]*depth
    rf[f"INT{neurons}"] = [0]*depth

    kmO = kerasMNIST(1)
    kmO.start(Breite=neurons)
    kmO.saveModel(m)

    W,b,activ,start,num = loadModel(kmO)

    for k in tqdm(range(depth),desc=f"Iteration {m + 1} of {width}"):
      
      inf = inferer()
      replace_dict = {
                    "clipped_relu" : inf.relu,
                    "relu" : inf.relu,
                    "softmax" : inf.softmax
      }      
      act = [*map(replace_dict.get,activ)]
      tbefore = time_ns()
      inf.forward_general(x,W,b,act,num-start)
      tafter = time_ns()
      df.at[k,f"INT{neurons}"] = (tafter - tbefore) / 1e3
      inf.evaluate(y)
      rf.at[k,f"INT{neurons}"] = inf.acc
      
      km = kerasMNIST(1)
      km.loadModel(m)
      df.at[k,f"GPU{neurons}"] = km.predict()/1e3
      rf.at[k,f"GPU{neurons}"] = km.eval()
      del km

      km = kerasMNIST(1)
      km.loadModel(m)
      df.at[k,f"CPU{neurons}"] = km.predict(GPU = False)/1e3
      rf.at[k,f"CPU{neurons}"] = km.eval()
      del km

  df.to_excel("Output_width_3.xlsx")
  rf.to_excel("Result_width_3.xlsx")