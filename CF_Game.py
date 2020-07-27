#!usr/bin/env python
#Engine for Connect N (default: N = 4)

import random
import numpy as np

class CF_Game:
	
	def __init__(self, game_id=0, n=4, rows=6, cols=7):
		self.game_id = game_id
		self.rows = rows
		self.cols = cols
		self.max_moves = rows*cols
		self.length = n 
		self.actions = cols
		self.reset()
	
	def reset(self, turn=1):
		self.count = 0
		self.game_over = False
		self.turn = turn
		self.board = [[0 for j in range(self.cols)] for i in range(self.rows)]
		self.peaks = [(self.rows - 1) for j in range(self.cols)]

	def is_game_over(self):
		return self.game_over

	def get_num_actions(self):
		return self.actions
	
	def print_board(self):
		temp = 0
		for i in range(self.rows):
			for j in range(self.cols):
				if self.board[i][j] == 0:
					temp = 0
				else:
					temp = 1 if self.board[i][j] == 1 else 2
				print(" |" + str(temp), end="")
			print("|")
		print(" ---------------------")

	def flip_board(self):
		for i in range(self.rows):
			for j in range(self.cols):
				self.board[i][j] *= - 1

	def get_board(self):
		return np.array(self.board).flatten().reshape((self.max_moves, 1))

	# next player's turn: Player 1, Player 2
	def get_turn(self):
		return self.turn

	def no_moves_left(self):
		return self.count == self.max_moves	

	def next_turn(self):
		self.turn *= -1

	def move(self, col):
		if self.is_game_over():
			return False
		if not self.is_valid_col(col):
			return False
		if self.peaks[col] >= 0:
			self.board[self.peaks[col]][col] = self.get_turn()
			self.peaks[col] -= 1
		else:
			return False

		self.next_turn()
		self.count += 1
		self.game_over = self.no_moves_left()
		return True
	
	#Check valid point dimensions

	def is_valid_row(self, row):
		return row >= 0 and row < self.rows
	
	def is_valid_col(self, col):
		return col >= 0 and col < self.cols

	def is_valid_coord(self, row, col):
		return self.is_valid_row(row) and self.is_valid_col(col)

	# Check for line of 1s or 2s   
	def check_for_line(self):
		found = True 

		height = self.rows - self.length + 1
		width = self.cols - self.length + 1

		#Row checking
		for i in range(height):
			for j in range(self.cols):
				curr = self.board[i][j]
				if curr == 0:
					continue
				for r in range(1, self.length):
					if self.board[i + r][j] != curr:
						found = False
						break
				if found:
					return curr 
				found = True

		#Column checking
		for i in range(self.rows):
			for j in range(width):
				curr = self.board[i][j]
				if curr == 0:
					continue
				for r in range(1, self.length):
					if self.board[i][j + r] != curr:
						found = False
						break
				if found:
					return curr
				found = True

		#Right Diagonal checking
		for i in range(height):
			for j in range(width):
				curr = self.board[i][j]
				if curr == 0:
					continue
				for r in range(1, self.length):
					if self.board[i + r][j + r] != curr:
						found = False
						break
				if found:
					return curr 
				found = True

		#Left Diagonal checking
		height = self.length - 1
		for i in range(height, self.rows):
			for j in range(width):
				curr = self.board[i][j]
				if curr == 0:
					continue
				for r in range(1, self.length):
					if self.board[i - r][j + r] != curr:
						found = False
						break
				if found: return curr 
				found = True
		return 0

	def get_board_state(self):
		result = self.check_for_line()
		if result != 0: self.game_over = True
		elif self.no_moves_left(): return 0.5
		return result

def main():
	cf = CF_Game(0)

	for i in range(10):
		cf.move(random.randint(0, 6))
		cf.print_board()
		print(cf.get_board_state())
		print(cf.get_board())
	# Do stuff

if __name__ == "__main__":
	main()
