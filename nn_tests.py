import unittest
import sys
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from monk_1 import Layer
import monk_1

df = pd.read_csv('monks-1.train', sep='\s+', skip_blank_lines=False, skipinitialspace=False)

df_enc = df.drop(['class', 'ID'], axis=1)

enc = OneHotEncoder()
enc.fit(df_enc)
df_ohe = enc.transform(df_enc).toarray()

df_class = df[['class']]
enc = OneHotEncoder()
enc.fit(df_class)
df_class = enc.transform(df_class).toarray()


class NnTests(unittest.TestCase):
    def forward_(self, n_in, n_hid, n_out, row):
        lay1 = Layer(n_in, n_hid)
        lay2 = Layer(n_hid, n_out)
        lay1.load_input(df_ohe[row], df_class[row])

        # create a dummy vector to test the algorithm
        dummy = np.zeros(lay1.n_units)
        for index, item in enumerate(dummy):
            dummy[index] = monk_1.net(lay1.weights[index], lay1.x)

        lay1.forward(lay2)

        self.assertAlmostEqual(dummy.tolist(), lay2.x.tolist())

    def test_forward_func(self):
        self.forward_(17, 4, 2, row=8)
        self.forward_(17, 8, 3, row=24)
        self.forward_(17, 7, 2, row=12)


if __name__ == '__main__':
    unittest.main()
