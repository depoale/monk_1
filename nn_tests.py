import unittest
import sys
import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from monk_1 import Layer
import monk_1
random.seed(42)

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
    def create(self, n_in, n_hid, n_out, row):
        lay1 = Layer(n_in, n_hid, n_out)
        lay2 = Layer(n_hid, n_out, n_out)
        lay1.load_input(df_ohe[row], df_class[row])
        return lay1, lay2

    def forward_(self, n_in, n_hid, n_out, row):
        lay1, lay2 = self.create(n_in, n_hid, n_out, row)

        # create a dummy vector to test the algorithm
        dummy = np.zeros(lay1.n_units)
        for index, item in enumerate(dummy):
            dummy[index] = monk_1.net(lay1.weights[index], lay1.x)

        lay1.forward(lay2)
        print('dummy', dummy)
        print(lay2.x)

        self.assertListEqual(dummy.tolist(), lay2.x.tolist())

    def forward_func(self):
        self.forward_(17, 4, 2, row=8)
        #self.forward_(17, 8, 3, row=24)
        #self.forward_(17, 7, 2, row=12)

    def delta_(self, n_in, n_hid, n_out, row):
        print(df_class)
        print('fine\n')
        final_class = []
        final_d = []
        lay1, lay2 = self.create(n_in, n_hid, n_out, row)
        for epoch in range(100):
            for pattern in range(len(df)):
                lay1.load_input(df_ohe[pattern], df_class[pattern])
                print(df_class[pattern])
                lay1.compute_output(monk_1.act_tanh)
                for i in lay1.out_vec:
                    self.assertTrue(-1 < i < 1.)
                lay1.forward(lay2)
                self.assertListEqual(lay1.out_vec.tolist(), lay2.x.tolist())
                lay2.compute_output(monk_1.act_ltu)
                for i in lay2.out_vec:
                    self.assertTrue(0. <= i <= 1.)

                lay2.evaluate_delta_output()
                #print(f'\n delta: {lay2.deltas}')
                lay2.evaluate_delta_partial_hidden(lay1)
                self.assertTrue(lay1.deltas.tolist() != 12.)
                lay1.evaluate_delta_hidden()
                for i in lay2.deltas:
                    self.assertTrue(-1. <= i <= 1.)

                lay2.update_weights()
                lay1.update_weights()
                if epoch == 100-1:
                    print(lay2.out_vec, lay2.d)
                    final_class.append(lay2.out_vec)
                    final_d.append(lay2.d)

            #print(final_class[0])

        for element in final_class:
            pass



    def test_delta(self):
        """a few tests using the function delta_"""
        self.delta_(17, 4, 2, row=8)
        #self.delta_(17, 8, 3, row=24)
        #self.delta_(17, 7, 2, row=12)

    def test_iter(self):
        """test iteration"""
        lay1, lay2 = self.create(17, 5, 2, row=7)
        x = np.linspace(0, (lay1.n_units)-1, lay1.n_units)
        for unit in lay1:
            self.assertAlmostEqual(unit, x[unit])



if __name__ == '__main__':
    unittest.main()
