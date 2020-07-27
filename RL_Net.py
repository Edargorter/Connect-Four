# Artificial Neural Network
# ANN skeleton constructed partly Using modified template code by Michael Nielsen (2018)
import numpy as np
import random
import time

class ANN:
	def __init__(self, layers):
		self.no_layers = len(layers)
		self.layers = layers
		self.construct_weights_biases()

	def construct_weights_biases(self):
			np.random.seed(int(time.time()))
			self.weights = [np.random.randn(i, j)/np.sqrt(j) for j, i in zip(self.layers[:-1], self.layers[1:])]
			self.biases = [np.random.randn(i, 1) for i in self.layers[1:]]

	def get_network_string(self):
		string = ""	
		for i in range(self.no_layers - 1):
			for j in range(self.layers[i + 1]):
				string += ','.join([str(self.weights[i][j][k]) for k in range(self.layers[i])])
				string += "b" + str(self.biases[i][j])
				if j != self.layers[i + 1] - 1:
					string += ";"
			if i != self.no_layers - 2:
				string += ":"
		return string

	def get_network(self):
		return self.net

	def get_weights(self):
		return self.weights
	
	def get_biases(self):
		return self.biases

	def set_weights(self, w):
		self.weights = w

	def set_biases(self, b):
		self.biases = b

	def sigmoid(self, z):
		return 1.0/(1.0 + np.exp(-np.clip(z, -500, 500)))

	def d_sigmoid(self, z):
		return self.sigmoid(z)*(1 - self.sigmoid(z))

	def linear(self, z):
		return z

	def d_linear(self, z):
		return 1.0

	def activation(self, z):
		return self.sigmoid(z)

	def deactivation(self, z):
		return self.d_sigmoid(z)

	def get_vector(self, dim, value):
		result = np.zeros((dim, 1))
		result[value] = 1.0
		return result

	def feed_forward(self, state):
		for b, w in zip(self.biases, self.weights):
			state = self.activation(np.dot(w, state) + b)
		return state

	def quadratic(self, y, predictions):
		return 0.5*np.linalg.norm(predictions - y)**2

	def quadratic_delta(self, z, a, y):
		return (a - y) * self.deactivation(z)

	def logistic(self, y, predictions):
		return np.sum(np.nan_to_num(-y*np.log(predictions) - (1 - y)*np.log(1 - predictions)))

	def logistic_delta(self, z, a, y):
		return (a - y)

	def back_propagate(self, x, y):
		dw = [np.zeros(w.shape) for w in self.weights]
		db = [np.zeros(b.shape) for b in self.biases]
		act = x
		activations = [x]
		zs = []

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, act) + b
			zs.append(z)
			act = self.activation(z)
			activations.append(act)

		delta = self.logistic_delta(zs[-1], activations[-1], y)
		dw[-1] = np.dot(delta, activations[-2].transpose())
		db[-1] = delta

		for l in range(2, self.no_layers):
			z = zs[-l]
			s = self.deactivation(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta)*s
			db[-l] = delta
			dw[-l] = np.dot(delta, activations[-l-1].transpose())

		return (db, dw)

	def update(self, x, y, l_rate, n):
		b = [np.zeros(bias.shape) for bias in self.biases]
		w = [np.zeros(weight.shape) for weight in self.weights]
		for x_i, y_i in zip(x, y):
			db, dw = self.back_propagate(x_i, y_i)
			b = [nb + dnb for nb, dnb in zip(b, db)]
			w = [nw + dnw for nw, dnw in zip(w, dw)]
		self.weights = [(1 - l_rate/n)*weight - (l_rate/n)*nw for weight, nw in zip(self.weights, dw)]
		self.biases = [bias - (l_rate/n)*nb for bias, nb in zip(self.biases, db)]

	def train_batch(self, x, y, l_rate):
		self.update(x, y, l_rate, 1)

def main(): # TESTING PURPOSES ONLY
	ann = ANN([10, 80, 7])
	a = np.random.randn(10, 1)*10
	y = np.random.randn(7, 1)*5
	w = ann.get_weights()
	b = ann.get_biases()
	ann.update(a, y, 0.1)

if __name__ == "__main__":
	main()
