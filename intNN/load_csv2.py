from os.path import exists
import json
from typing import Callable
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from numpy import int8,int16,int32

class load_csv():
    def __init__(self) -> None:
         pass

    def load_general(self, start: int, rel_path: str, clipped_relu: Callable, relu: Callable, softmax: Callable, dtypeNormMax: int, dtypeNORM: np.dtype, ending = "txt",weights_prefix = "W", bias_prefix = "b", xtest_name = "xtest", ytest_name = "ytest", index_col = None,header=None) -> tuple[list, list, NDArray, NDArray[int8], list, int, int]:
        """Load weights, biases and test data.

        Args:
            start (int): Layer count start (in most cases 1)
            rel_path (str): Folder in which to find the files
            clipped_relu (callback): clipped relu function
            relu (callback): relu function
            softmax (callback): softmax function
            ending (str, optional): File ending. Defaults to "txt".
            weights_prefix (str, optional): Prefix in front of the weights file(s). Defaults to "W".
            bias_prefix (str, optional): Prefix in front of the bias file(s). Defaults to "b".
            xtest_name (str, optional): Name of the Test Input file. Defaults to "xtest".
            ytest_name (str, optional): Name of the Test Target file. Defaults to "ytest".

        Returns:
            W (list): List of numpy.ndarray weight arrays
            b (list): List of numpy.ndarray bias arrays
            xtest (numpy.ndarray): X Test values
            ytest (numpy.ndarray): Y Test values
            activation (list): List of callback functions for each layer
            start_layer (int): Number of first Layer
            num_layers (int): Number of last Layer
        """        
        self.data_imported = True
        i = start
        while exists(f"{rel_path}/W{i}.{ending}"):
            setattr(self,f"W{i}",(pd.read_csv(f"{rel_path}/{weights_prefix}{i}.{ending}",index_col=index_col,header=header).to_numpy() * dtypeNormMax).astype(dtypeNORM))
            setattr(self,f"b{i}",(pd.read_csv(f"{rel_path}/{bias_prefix}{i}.{ending}",index_col=index_col,header=header).to_numpy() * dtypeNormMax).astype(dtypeNORM))
            setattr(self,f"b{i}",np.reshape(getattr(self,f"b{i}"),(getattr(self,f"b{i}")).shape[0]))

            i += 1
        
        self.xtest = (pd.read_csv(f"{rel_path}/{xtest_name}.{ending}",index_col=index_col,header=header).to_numpy()).astype(dtypeNORM)
        self.ytest = (pd.read_csv(f"{rel_path}/{ytest_name}.{ending}",index_col=index_col,header=header).to_numpy().astype(np.int8))

        self.num_layers = i
        self.start_layer = start

        with open(f"{rel_path}/model.json","r") as data:
            self.keras_settings = json.load(data)
        
        replace_dict = {
                         "clipped_relu" : clipped_relu,
                         "relu" : relu,
                         "softmax" : softmax
        }

        self.W = []
        self.b = []
        self.activation = []
        for i in range(self.start_layer,self.num_layers):
            self.activation.append(replace_dict[self.keras_settings["config"]["layers"][i]["config"]["activation"]])
            self.W.append(getattr(self,f"W{i}"))
            self.b.append(getattr(self,f"b{i}"))

        return self.W, self.b, self.xtest, self.ytest, self.activation, self.start_layer, self.num_layers