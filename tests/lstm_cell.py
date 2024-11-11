import numpy as np
import tensorflow as tf
from keras.api.layers import LSTM

tf.random.set_seed(0)


def sigmoid_fn(Z: np.ndarray):
    return 1 / (1 + np.exp(-Z))


def tan_h_fn(Z: np.ndarray):
    e_x = np.exp(Z)
    e_neg_x = np.exp(-Z)
    return (e_x - e_neg_x) / (e_x + e_neg_x)


class LSTMCell:
    def __init__(self, w_i, w_f, w_c, w_o):
        self.current_long = np.array([0]).reshape(-1, 1)
        self.current_short = np.array([0]).reshape(-1, 1)
        self.forget_gate_weights = w_f
        self.input_gate_weights = w_i
        self.cc_gate_weights = w_c
        self.output_gate_weights = w_o

    @property
    def short_term_memory(self):
        return self.current_short

    @property
    def long_term_memory(self):
        return self.current_long

    def run(self, X: np.ndarray):
        X = np.stack([self.current_short, X]).reshape(-1, 1)
        forget_gate = sigmoid_fn(X.T.dot(self.forget_gate_weights))
        long_memory = forget_gate * self.current_long
        input_gate = sigmoid_fn(X.T.dot(self.input_gate_weights))
        current_cell_state_gate = tan_h_fn(X.T.dot(self.cc_gate_weights))
        long_memory += input_gate * current_cell_state_gate
        og = tan_h_fn(long_memory) * sigmoid_fn(X.T.dot(self.output_gate_weights))
        self.current_long = long_memory
        self.current_short = og
        return og


def main():
    keras_lstm = LSTM(units=1, use_bias=False)
    X = np.array([0.78]).reshape(-1, 1)
    keras_response = np.asarray(keras_lstm(X.reshape(-1, 1, 1)))
    w_i, w_f, w_c, w_o = tf.split(keras_lstm.weights[0], 4, axis=1)
    one = np.array([1]).reshape(-1, 1)
    w_i = np.asarray(w_i)
    w_f = np.asarray(w_f)
    w_c = np.asarray(w_c)
    w_o = np.asarray(w_o)

    w_i = np.hstack((one, w_i)).T
    w_f = np.hstack((one, w_f)).T
    w_o = np.hstack((one, w_o)).T
    w_c = np.hstack((one, w_c)).T

    print(keras_response)
    lstm = LSTMCell(w_i, w_f, w_c, w_o)
    lstm.run(X=X)
    print(lstm.current_short)


if __name__ == '__main__':
    main()
