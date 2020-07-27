import numpy as np

class TTT:
	def __init__(self, game_id, n=3, rows=3, cols=3):
		self.max_moves = rows * cols
		self.rows = rows
		self.cols = cols
		self.reset()
		self.length = n
		self.actions = rows*cols
	
	def reset(self, turn=1):
		self.count = 0
		self.game_over = False
		self.turn = turn
		self.board = [0 for i in range(self.max_moves)]

	def is_game_over(self):
		return self.game_over

	def get_turn(self):
		return self.turn

	def get_num_actions(self):
		return self.actions

	def print_board(self):
		temp = 0
		for i in range(self.rows):
			for j in range(self.cols):
				if self.board[i*self.cols + j] == 0:
					temp = 0
				else:
					temp = 1 if self.board[i*self.cols + j] == 1 else 2
				print(" |" + str(temp), end="")
			print("|")
		print(" ---------------------")

	def get_board(self):
		return np.array(self.board).reshape((self.max_moves, 1))

	def flip_board(self):
		for i in range(self.rows):
			for j in range(self.cols):
				self.board[i][j] *= -1

	# next player's turn: Player 1, Player 2
	def get_turn(self):
		return self.turn

	def no_moves_left(self):
		return self.count == self.max_moves	

	def next_turn(self):
		self.turn *= -1

	def valid(self, move):
		return move >= 0 and move < self.max_moves

	def move(self, move):
		if self.is_game_over(): return False
		if not self.valid(move): return False

		if self.board[move] == 0:
			self.board[move] = self.get_turn()
			self.next_turn()
			self.count += 1
			self.game_over = self.no_moves_left()
			return True

		return False

	def check_for_line(self):
		for turn in [-1,1]:
			for offset in range(3):
				# Col checking
				if self.board[offset] == self.board[3 + offset] == self.board[6 + offset] == turn:
					return turn
				# Row checking
				if self.board[offset * self.cols] == self.board[1 + offset * self.cols] == self.board[2 + offset * self.cols] == turn:
					return turn

			# Diagonal checking
			if self.board[0] == self.board[4] == self.board[8] == turn:
				return turn
			if self.board[2] == self.board[4] == self.board[6] == turn:
				return turn
		return 0

	def get_board_state(self):
		result = self.check_for_line()
		if result != 0: self.game_over = True
		return result
