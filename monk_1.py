# -*- coding: utf-8 -*-
"""Monk_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U_MiiIKuRphcbJ_xKi552Bsa-Dng31jf
"""

#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import random 
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('monks-1.train', sep='\s+', skip_blank_lines=False, skipinitialspace=False)

df_enc = df.drop(['class', 'ID'], axis=1)
print()

enc = OneHotEncoder()
enc.fit(df_enc)
df_ohe = enc.transform(df_enc).toarray()

df_class = df[['class']]
enc = OneHotEncoder()
enc.fit(df_class)
df_class = enc.transform(df_class).toarray()


L = len(df)
ETA = 10e-4/L
ALPHA = 0.5
LAMBDA = 0.001
N_UNITS = 3
N_EPOCHS = 500
ERROR = 1000
random.seed(42)

def net(w, x):
  return np.dot(w, x)

def act_func(x, alpha):
  return np.tanh(alpha*x/2)

def act_ltu(x):
  return np.heaviside(x,0)

def act_derivative(x, alpha):
  return (1-(np.tanh(alpha*x/2))**2)

class Layer:

  ''' single layer '''

  def __init__(self, n_inputs, n_units):

    self.n_inputs = n_inputs
    self.n_units = n_units

    self.weights = np.random.uniform(-0.7, 0.7, size=(n_units, n_inputs))

    self.x = np.zeros(n_inputs)
    self.d = 0
    self.deltas = np.zeros(n_units)

  def load_input(self, input_values: npt.ArrayLike, target):
    self.x = input_values
    self.d = target

  def forward(self, next_l):
    out_vec = np.full(self.n_units, 12.)
    for i in range(len(self.weights)):
      out_vec[i] = net(self.weights[i], self.x)
    next_l.x = out_vec


  def evaluate_delta_hidden(self, next_delta):
    pass



  def evaluate_delta_output(self):
    ''' delta output layer'''
    
    for j in range(2):
      for i in range(len(self.x)-1):
       print(self.d)
       print(self.x[i])
       self.deltas[j] = (self.d[j] - self.x[i])* act_derivative(self.x[i], ALPHA)
'''

df

hidden = Layer(17,N_UNITS)
output_l= Layer(N_UNITS,2)
hidden.load_input(df_ohe[8], df_class[8])
hidden.forward(output_l)
print(output_l.x)
output_l.evaluate_delta_output()
'''