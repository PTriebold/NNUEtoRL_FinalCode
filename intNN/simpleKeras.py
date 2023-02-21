import numpy as np
import keras
from keras import backend as K
import mnist
from keras.utils import to_categorical
import general2
import pandas as pd
from tqdm import tqdm
from typing import Any


class simpleKeras():
    def __init__(self, lr = 1e-3, test_size = 20, batch = 65, epochs = 15) -> None:
        self.num_classes = 10
        self.input_shape = 784

        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = lr

        self.xtrain = mnist.train_images().reshape((-1,784))
        self.ytrain = to_categorical(mnist.train_labels())
        self.xtest = mnist.test_images().reshape((-1,784)).astype(np.uint8)
        self.ytest = to_categorical(mnist.test_labels())
        
        self.input, self.target = (self.xtest,self.ytest)


    def clipped_relu(self, x) -> Any:
        """Clipped relu activation function

        Args:
            x (list|array|tensor): Input data

        Returns:
            Tensor: data after activation function
        """
        return K.relu(x, max_value=1)

    def makeModel(self):
        """Build the model
        """
        self.model = keras.Sequential()
        self.model.add(Dense(256,activation=self.clipped_relu))
        self.model.add(Dense(128,activation=self.clipped_relu))
        self.model.add(Dense(self.num_classes, activation="softmax"))
        
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train(self):
        """Train the model
        """
        self.model.fit(self.xtrain, self.ytrain, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1,verbose=0)

    def predict(self):
        """Predict data
        """
        self.prediction = self.model.predict(self.xtest, verbose=0)

    def eval(self,out = False):
        """Evaluate the prediction results

        Args:
            out (bool, optional): Print to console. Defaults to False.
        """
        self.score = np.count_nonzero(np.argmax(self.prediction,axis=1) == np.argmax(self.ytest,axis=1)) / self.ytest.shape[0]
        self.pred = np.argmax(self.prediction,axis=1)
        self.real = np.argmax(self.ytest,axis=1)
        if out: print("Test accuracy:", self.score)

    def save(self):
        """Save to disk
        """
        for i, layer in enumerate(self.model.layers):
            weigths = layer.get_weights()
            pd.DataFrame(weigths[0]).to_csv(f"Kout/W{i+1}.txt",index=False,header=False)
            pd.DataFrame(weigths[1]).to_csv(f"Kout/b{i+1}.txt",index=False,header=False)
            
        pd.DataFrame(self.target).to_csv("Kout/ytrain.txt",index=False,header=False)
        pd.DataFrame(self.input).to_csv("Kout/xtrain.txt",index=False,header=False)
        pd.DataFrame(self.ytest).to_csv("Kout/ytest.txt",index=False,header=False)
        pd.DataFrame(self.xtest).to_csv("Kout/xtest.txt",index=False,header=False)

        with open("Kout/model.json","w") as file:
            file.writelines(self.model.to_json())

def complete(learning_rate = 1e-3, test_size = 20, batch = 65, epochs = 15) -> float:
    """Helper function to run the complete keras script

    Args:
        learning_rate (float, optional): Learning rate. Defaults to 1e-3.
        test_size (int, optional): how many data should be tested. Defaults to 20.
        batch (int, optional): Batch size. Defaults to 65.
        epochs (int, optional): N epochs. Defaults to 15.

    Returns:
        float: Accuracy score
    """
    sk = simpleKeras(lr = learning_rate, test_size = test_size, batch = batch, epochs = epochs)
    sk.makeModel()
    sk.train()
    sk.predict()

    sk.eval(out=True)
    sk.save()

    return sk.score

if __name__ == "__main__":
    RUNS = 15
    results = np.zeros((RUNS,2)) 
    for i in tqdm(range(RUNS)):
        a = complete(learning_rate = 1e-3, test_size = 20, batch = 65, epochs = 15)
        b = general2.gesamt()
        results[i,0] = a
        results[i,1] = b


    print(results)