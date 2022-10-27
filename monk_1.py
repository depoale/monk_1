# -*- coding: utf-8 -*-
"""Monk_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U_MiiIKuRphcbJ_xKi552Bsa-Dng31jf
"""

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import random
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder
from utils import net_fun, act_tanh, act_ltu, derivative_tanh

df = pd.read_csv('monks-1.train', sep='\s+', skip_blank_lines=False, skipinitialspace=False)

L = len(df)
ETA = 10e-2 / L
ALPHA = 0.5
LAMBDA = 0.001
N_UNITS = 3
N_EPOCHS = 500
ERROR = 1000
random.seed(42)

df_enc = df.drop(['class', 'ID'], axis=1)
print()

enc = OneHotEncoder()
enc.fit(df_enc)
df_ohe = enc.transform(df_enc).toarray()

df_class = df[['class']]
enc = OneHotEncoder()
enc.fit(df_class)
df_class = enc.transform(df_class).toarray()

class Layer:
    ''' single layer (no distiction btw hidden and output layer)'''

    def __init__(self, n_inputs, n_units, class_num=2):

        self.n_inputs = n_inputs
        self.n_units = n_units
        self.class_num = class_num

       #initialize weights using fan-in inverse
        fan_in = n_inputs
        self.weights = np.random.uniform(-1/fan_in, 1/fan_in, size=(n_units, n_inputs))

        self.x = np.zeros(n_inputs)  # layer inputs
        self.d = [0, 1]  # final target (forse dobbiamo inizializzarlo a tuple)
        self.deltas = np.full(self.n_units, 12.)
        # layer delta array (initialised as zero, overwritten w/ backprop algorithm)
        self.out_vec = np.full(self.n_units, 12.)
        self.net = np.full(self.n_units, 12.)

    def __iter__(self):
        return iter(range(self.n_units))

    def load_input(self, input_values: npt.ArrayLike, target):
        """
        Used to load one hot encoded data onto 1st hidden layer as input.
        We save the target values aswell.

        Parameters
        ---------

        input_values: scalar or array-like
        target: scalar or array-like

        Raises
        ------
        TypeError
            If input_values or target are not array-like

        IndexError
            If number of expected inputs and actual inputs of units don't match

        IndexError
            If target size and output size don't match
        """
        if not (type(input_values) or type(target)) is np.ndarray:
            raise TypeError(f'Inputs and target must be array-like')

        if len(self.x) != len(input_values):
            raise IndexError(f'Input dimension and length layer.x dont match')

        if len(self.d) != self.class_num:
            raise IndexError(f'Target size and output size dont match')

        self.x = input_values
        self.d = target.astype(float)

    def compute_output(self, activation=act_tanh) -> npt.NDArray:
        """
        Create and compute out_vec (shape=n_units)

        Parameters
        ----------
        activation: activation function, default act_tanh
        """
        for i in range(self.n_units):
            self.net[i] = net_fun(self.weights[i], self.x)  # net[i] is the ith-unit net
        self.out_vec = activation(self.net)

    def forward(self, next_l):
        '''Feedforward NN function. Evaluate current layer outputs and set them as next layer inputs
        Parameters
        --------
        next_l: Layer-like
            Next layer in the NN

        Raises
        ------
        IndexError
            If the dimensions of current layer units and next layer input don't match
        '''

        if self.n_units != next_l.n_inputs:
            raise IndexError(f'Cant compute if dim {self.n_units} and {next_l.n_inputs} dont match')

        next_l.x = self.out_vec  # we use act_tanh (not act_ltu) bc this is used only by hidden layers

    def evaluate_delta_partial_hidden(self, previous_layer):
        """ Evaluate partially delta hidden layer
        (only sum over k (w_kj \cdot delta_k)) """
        partial_delta = np.zeros(previous_layer.n_units)
        for j in range(previous_layer.n_units):
            for i in range(self.n_units):
                partial_delta[j] += self.weights[i][j]*self.deltas[i]

        previous_layer.deltas = partial_delta

    def evaluate_delta_hidden(self):
        """using partial_delta, evaluate the final delta for backprop
        remember: self.deltas= partial_delta computed by the next layer in previous step"""
        self.deltas = self.deltas * derivative_tanh(self.net)


    def evaluate_delta_output(self):
        ''' delta output layer for backprop'''
        for j in range(self.n_units):
            self.deltas[j] = (self.d[j] - self.out_vec[j]) * derivative_tanh(self.net[j], ALPHA)


    def update_weights(self):
        '''
        Update weights using on-line gradient descend
        '''

        for i in range(self.n_units):
            for j in range(len(self.x)):   #range(len(self.x))=previous_layer.n_units
                #print(f'previous: {self.weights[i][j]}')
                self.weights[i][j] += ETA*self.deltas[i]*self.x[j]-LAMBDA*self.weights[i][j]
                #print(f'current: {self.weights[i][j]}')

if __name__ == '__main__':
    hidden = Layer(17, N_UNITS)
    output_l = Layer(N_UNITS, 2)
    hidden.load_input(df_ohe[8], df_class[8])
    hidden.forward(output_l)
    output_l.evaluate_delta_output()
    print(np.shape(output_l.deltas))
