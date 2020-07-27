# An engine for Simple Tak : Zachary Bowditch (2019)
import random

class ST:
	def __init__(self, dim, board=None):
		self.dim = dim
		self.moves = [[(0,1), (1,0), (-1,0)], [(1,0), (0,1), (0,-1)]]
		self.max_moves = self.dim * self.dim
		self.reset(board)

	def reset(self, board):
		if board != None:
			self.board = board.copy()
		else:
			self.board = [[0 for j in range(self.dim)] for i in range(self.dim)]

		self.turn = 1
		self.count = 0

	def print_board(self):
		for i in range(self.dim):
			for j in range(self.dim):
				print(self.board[i][j], end=" ")
e		print("\n")

	def valid_coord(self, r, c):
		return r >= 0 and r < self.dim and c >= 0 and c < self.dim
		
	def move(self, r, c):
		if not self.valid_coord(r, c):
			return False
		if self.board[r][c] != 0:
			return False
		self.board[r][c] = self.turn 
		self.turn = self.turn % 2 + 1
		self.count += 1
		return True

	def DFS(self, origin, player, r, c, visited):
		if origin:
			if r == self.dim - 1: return True
		else:
			if c == self.dim - 1: return True
		visited[r][c] = True	
		for i in range(len(self.moves)):
			row, col = r + self.moves[origin][i][0], c + self.moves[origin][i][1]
			if self.valid_coord(row, col):
				if self.board[row][col] == player and not visited[row][col]:
					return self.DFS(origin, player, row, col, visited)
		return False
				

	def evaluate_board(self):
		if self.count == self.max_moves: return -1

		visited = [[False for j in range(self.dim)] for i in range(self.dim)]

		# Horizontal	

		for i in range(self.dim):
			player = self.board[i][0]
			if player == 0: continue
			if self.DFS(False, player, i, 0, visited): return player 

		# Vertical

		for i in range(self.dim):
			player = self.board[0][i]
			if player == 0: continue
			if self.DFS(True, player, 0, i, visited): return player 

		return 0

def main():
	board = [[2,2,0,0],[0,2,0,0],[0,0,2,2],[0,0,0,0]]
	st = ST(4, board)
	st.print_board()
	print(st.evaluate_board())

if __name__ == "__main__":
	main()
