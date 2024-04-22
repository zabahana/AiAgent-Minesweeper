import numpy as np

class QLearningAgent:

    def __init__(self, board_size, learning_rate=0.001, discount_factor=0.99):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = np.zeros((board_size, board_size, 3))

    def take_action(self, state, board):
        # Choose the action with the highest Q-value for the current state.
        if board[state] > 0:  # If the cell has a number indicating adjacent mines
            # Check if the number of adjacent unrevealed cells matches the number of adjacent mines
            num_adjacent_unrevealed = np.sum(board[max(state[0] - 1, 0):min(state[0] + 2, self.board_size), max(state[1] - 1, 0):min(state[1] + 2, self.board_size)] == 0.5)
            if num_adjacent_unrevealed == board[state]:
                # If yes, mark all adjacent unrevealed cells as mines
                for i in range(max(state[0] - 1, 0), min(state[0] + 2, self.board_size)):
                    for j in range(max(state[1] - 1, 0), min(state[1] + 2, self.board_size)):
                        if board[i, j] == 0.5:
                            return 1, (i, j)  # Mark square as a mine
        # Otherwise, choose the action with the highest Q-value
        action = np.argmax(self.q_values[state])
        return action, state

    def update_q_values(self, state, action, reward, next_state):
        # Update the Q-value for the current state-action pair.
        current_q_value = self.q_values[state][action]
        next_q_value = np.max(self.q_values[next_state])
        updated_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value - current_q_value)
        self.q_values[state][action] = updated_q_value

def build_board():
    # Create a 10x10 board with numbers indicating adjacent mines.
    board = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if np.random.random() < 0.1:
                board[i][j] = -1  # Mine
                # Update adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= i + dx < 10 and 0 <= j + dy < 10 and board[i + dx][j + dy] != -1:
                            board[i + dx][j + dy] += 1
    return board

def print_board(board, uncovered_cell=None):
    # Print the board
    print("Current Board:")
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if (i, j) == uncovered_cell:  # Uncovered cell
                print("U", end=" ")
            elif board[i][j] == -1:  # Mine
                print("M", end=" ")
            elif board[i][j] == 0.5:  # Uncovered safe square
                print(".", end=" ")
            elif board[i][j] == -0.5:  # Marked as mine
                print("X", end=" ")
            else:  # Number indicating adjacent mines
                print(int(board[i][j]), end=" ")
        print()

def play_game(agent, board, max_moves=100):
    # Play the game until it is finished.
    state = (0, 0)
    total_reward = 0
    moves = 0
    
    # Generate a list of coordinates in random order
    all_coordinates = [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1])]
    np.random.shuffle(all_coordinates)

    for coordinate in all_coordinates:
        if moves >= max_moves:
            print("Maximum moves reached. Game over.")
            break
            
        # Uncover the box at the current coordinate
        state = coordinate
        
        # Skip if already uncovered
        if board[state] == 0.5:
            continue
            
        action, next_state = agent.take_action(state, board)
        if action == 0:  # Uncover square
            if board[state] == -1:  # Mine
                print("Game over! You hit a mine.")
                break
            else:
                reward = 1
                # Mark the square as uncovered.
                board[state] = 0.5
                uncovered_cell = state
        elif action == 1:  # Mark square as a mine
            reward = 0.8  # Give a smaller positive reward for marking a square as a mine.
            # Mark the square as a mine.
            board[state] = -0.5
            uncovered_cell = None
        else:  # Move to an uncovered square.
            reward = 0
            # Find an adjacent uncovered square.
            adjacent_squares = [(state[0] + dx, state[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if 0 <= state[0] + dx < board.shape[0] and 0 <= state[1] + dy < board.shape[1]]
            for adjacent_square in adjacent_squares:
                if board[adjacent_square] == 0.5:
                    state = adjacent_square
                    uncovered_cell = state
                    break

        total_reward += reward
        # Update the Q-values based on the current state, action, reward, and next state.
        agent.update_q_values(state, action, reward, next_state)

        # Print reward for each iteration
        print("Reward for current iteration:", reward)

        # Print the board after each iteration
        print_board(board, uncovered_cell)

        # Print total reward for each iteration
        print("Total Reward for current iteration:", total_reward)

        moves += 1

# Create a Q-learning agent.
agent = QLearningAgent(board_size=10)

board = build_board()

play_game(agent, board)
