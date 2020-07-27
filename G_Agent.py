# Train the Q Rewards using ANN
import numpy as np
import random
from RL_Net import ANN 

class G_Agent:
	def __init__(self, layers, actions, l_rate=0.01, discount=0.95, explore=0.9, iterations=1000):
		self.reset(layers, l_rate, discount, explore, iterations)
		self.actions = actions - 1

	def reset(self, layers, l_rate, discount, explore, iterations):
		self.l_rate = l_rate
		self.discount = discount
		self.explore = explore
		self.explore_delta = 1.0 / iterations
		self.iterations = iterations
		self.create_model(layers)

	def decrease_exploration(self):
		if self.explore > 0: self.explore -= self.explore_delta	

	def get_explore(self):
		return self.explore

	def set_l_rate(self, new_l_rate):
		self.l_rate = new_l_rate

	def get_rewards(self, state):
		return self.model.feed_forward(state)	

	def create_model(self, layers):
		self.model = ANN(layers)

	def get_model(self):
		return self.model.get_network()

	def set_model_params(self, w, b):
		self.model.set_weights(w)
		self.model.set_biases(b)

	def get_model_params(self):
		return (self.model.get_weights(), self.model.get_biases())

	def get_model_string(self):
		return self.model.get_network_string()

	def get_random_action(self):
		return random.randint(0, self.actions)
	
	def softmax(self, vector):
		exp = np.exp(vector)
		return exp/np.sum(exp)

	def get_action(self, state):
		return self.get_rewards(state)

	def train_online(self, old_state, action, reward, new_state):
		rewards = self.get_rewards(old_state)
		new_rewards = self.get_rewards(new_state)
		rewards[action] = reward + self.discount * np.amax(new_rewards)
		self.model.update(old_state, rewards, self.l_rate)

	def update_model(self, states, actions):
		self.model.train_batch(states, actions, self.l_rate)
