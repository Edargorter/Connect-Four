# Trains two different agents against each other
# Using Deep-Q learning ...
from G_Agent import G_Agent
from CF_Game import CF_Game
from TTT_Game import TTT
from sys import argv
import random
import time
import numpy as np

def export_net(net, filename):
	with open(filename, 'w') as f:
		f.write(net)

# Flip 1 to -1 and vice versa for player 2 input
def flip_board(board):
	return -1*board

if len(argv) < 2:
	print("Usage: python3 {} [ ID ]".format(argv[0]))
	exit(1)

game_id = int(argv[1])
l_rate = 0.1
discount = 0.95
explore = 0.5

#Reward values
victory = 100
defeat = -100
draw = 0

rows, cols = 3, 3
n = 3
iterations = game_id
period = 50
num_games = 100

rg = TTT(game_id, n)
tg = TTT(game_id, n)
#rg = CF_Game(game_id)
#tg = CF_Game(game_id)

num_actions = rg.get_num_actions()  # up to N possible actions for either player

layers = [rows*cols, 100, 80, 50, num_actions]
agent = G_Agent(layers, num_actions, l_rate, discount, explore, iterations)
target_agent = G_Agent(layers, num_actions, l_rate, discount, explore, iterations)

# Test board 
'''
for i in range(20): tg.move(random.randint(0, num_actions - 1))
tg.print_board()
tg.flip_board()
tg.print_board()
exit(1)
'''

def play_random_game(agent):
	victories = [0, 0, 0]
	start = 1
	result = 0
	for g in range(num_games):
		turn = start
		rg.reset(start)
		while(not rg.is_game_over()):
			if rg.get_turn() == 1:
				options = target_agent.get_action(rg.get_board())
				move = np.argmax(options)
				if not rg.move(move): 
					victories[1] += 1
					break
			else:
				move = random.randint(0, num_actions - 1)
				while not rg.move(move): move = (move + 1) % num_actions

			result = rg.get_board_state()

		if result == 1: victories[0] += 1
		elif result == -1: victories[1] += 1
		elif result == 0.5: victories[2] += 1
		start *= -1
	return victories

def train():
	global l_rate

	print_stuff = True

	prev_win_rate = 0

	f = open("TTT_(valid moves)_progress_net_100_80_50_{}.txt".format(game_id), 'w')

	five_win_rate = [0,0,0,0,0]
	five_loss_rate = [0,0,0,0,0]
	five_draw_rate = [0,0,0,0,0]

	start = time.time()

	p1_rewards = []
	p1_states = []
	p1_action_values = []
	p1_actions = []

	p2_rewards = []
	p2_states = []
	p2_action_values = []
	p2_actions = []

	game_states = []
	game_rewards = []

	for game in range(1, iterations):

		tg.reset() # Start new game

		while 1:
			valid_move = True
			state = tg.get_board()

			if tg.get_turn() == 1: 
				action_values = target_agent.get_action(state) # Player 1

				if random.random() > target_agent.get_explore():
					action = random.randint(0, num_actions - 1)
				else:
					action = np.argmax(action_values)
				
				if not tg.move(action): valid_move = False

				#while not tg.move(action): action = (action + 1) % num_actions # Change action until valid move is available 

				p1_states.append(state)
				p1_action_values.append(action_values)
				p1_actions.append(action)
				p1_rewards.append(np.array([action_values[action]]))

			else:
				action_values = target_agent.get_action(flip_board(state)) # Player 2

				if random.random() > target_agent.get_explore():
					action = random.randint(0, num_actions - 1)
				else:
					action = np.argmax(action_values)

				if not tg.move(action): valid_move = False

				#while not tg.move(action): action = (action + 1) % num_actions # Change action until valid move is available
				 
				p2_states.append(flip_board(state))
				p2_action_values.append(action_values)
				p2_actions.append(action)
				p2_rewards.append(np.array([action_values[action]]))

			if valid_move:
				result = tg.get_board_state()

				if result == 1: # Victory Player 1
					reward_1 = victory
					reward_2 = defeat

					p1_rewards[-1] = np.array([reward_1])
					p2_rewards[-1] = np.array([reward_2])
					
				elif result == -1: # Victory Player 2
					reward_1 = defeat
					reward_2 = victory

					p1_rewards[-1] = np.array([reward_1])
					p2_rewards[-1] = np.array([reward_2])

				elif result == 0.5: # Draw
					reward_1 = draw
					reward_2 = draw

					p1_rewards[-1] = np.array([reward_1])
					p2_rewards[-1] = np.array([reward_2])
			else:
				if tg.get_turn() == 1:
					p1_rewards[-1] = defeat
				else:
					p2_rewards[-1] = defeat
				break

			if tg.is_game_over(): break	
		
		# Get discounted reward sequences	

		p1_action_values[-1][p1_actions[-1]] = p1_rewards[-1] # player 1

		for i in range(len(p1_actions) - 2, -1, -1):
			p1_action_values[i][p1_actions[i]] = l_rate*(p1_rewards[i] + np.amax(p1_rewards[i + 1])*discount - p1_action_values[i][p1_actions[i]]) + p1_action_values[i][p1_actions[i]]

		p2_action_values[-1][p2_actions[-1]] = p2_rewards[-1] # player 2

		for i in range(len(p2_actions) - 2, -1, -1):
			p2_action_values[i][p2_actions[i]] = l_rate*(p2_rewards[i] + np.amax(p2_rewards[i + 1])*discount - p2_action_values[i][p2_actions[i]]) + p2_action_values[i][p2_actions[i]]

		states = p1_states + p2_states
		action_rewards = p1_action_values + p2_action_values

		game_states += states
		game_rewards += action_rewards

		if print_stuff and game % period == 0:

			# Train model based on accumulated state / reward pairs 
			agent.update_model(game_states, game_rewards)

			game_states = []
			game_rewards = []

			w, b = agent.get_model_params()
			print()
			print(w[0][0])
			print()
			target_agent.set_model_params(w, b)

			# Play games vs random agent to monitor improvement 
			victories = play_random_game(target_agent)
			summ = sum(victories)

			win_rate = round(100 * victories[0] / summ, 2)
			loss_rate = round(100 * victories[1] / summ, 2)
			draw_rate = round(100 * victories[2] / summ, 2)

			if win_rate > prev_win_rate:
				l_rate *= 0.5
				agent.set_l_rate(l_rate)

			prev_win_rate = win_rate

			five_win_rate.pop(0) 
			five_win_rate.append(win_rate)

			five_loss_rate.pop(0)
			five_loss_rate.append(loss_rate)
			
			five_draw_rate.pop(0)
			five_draw_rate.append(draw_rate)

			avg_win_rate = round(sum(five_win_rate)/5, 2)
			avg_loss_rate = round(sum(five_loss_rate)/5, 2)
			avg_draw_rate = round(sum(five_draw_rate)/5, 2)

			percent = round(100 * game / iterations, 2)
			string = "Status: {}% Wins: {}% ({}%) Losses: {}% ({}%) Draws: {}% ({}%)".format(percent, win_rate, avg_win_rate, loss_rate, avg_loss_rate, draw_rate, avg_draw_rate)
			f.write(string + "\n")
			print(string)
	#		print("Explore: {}".format(target_agent.get_explore()))

#		agent.decrease_exploration()
		target_agent.decrease_exploration()

	f.close() # Close file

	finish = time.time() - start
	print("Completed in {:.2f} s".format(finish))
	export_net(agent.get_model_string(), "agent_{}{}_{}_net.txt".format(rows, cols, game_id))

if __name__ == "__main__":
	train()
