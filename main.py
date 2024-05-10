from matplotlib import pyplot as plt
import numpy as np

class QLearningAgent:

    def __init__(self, board_size, learning_rate=0.001, discount_factor=0.99, initial_epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.999):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = np.zeros((board_size, board_size, 3))
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def take_action(self, state, board):
        # Choose the action with the highest Q-value for the current state.
        if board[state] > 0 and np.random.rand() < self.epsilon:  # If the cell has a number indicating adjacent mines
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
    
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_q_values(self, state, action, reward, next_state):
        # Update the Q-value for the current state-action pair.
        current_q_value = self.q_values[state][action]
        next_q_value = np.max(self.q_values[next_state])
        updated_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value - current_q_value)
        self.q_values[state][action] = updated_q_value

def build_board(board_size):
    # Create a board with numbers indicating adjacent mines.
    board = np.zeros((board_size, board_size))
    for i in range(board_size):
        for j in range(board_size):
            if np.random.random() < 0.1:
                board[i][j] = -1  # Mine
                # Update adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= i + dx < board_size and 0 <= j + dy < board_size and board[i + dx][j + dy] != -1:
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

def play_game(agent, board, board_size, max_moves=100):
    # Play the game until it is finished.
    state = (0, 0)
    total_reward = 0
    moves = 0
    
    # Generate a list of coordinates in random order
    all_coordinates = [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1])]
    np.random.shuffle(all_coordinates)

    for coordinate in all_coordinates:
        if moves >= max_moves:
            break
            
        # Uncover the box at the current coordinate
        state = coordinate
        
        # Skip if already uncovered
        if board[state] == 0.5:
            continue
            
        action, next_state = agent.take_action(state, board)
        if action == 0:  # Uncover square
            if board[state] == -1:  # Mine
                #print("Game over! You hit a mine.");
                break
            else:
                reward = 1
                # Mark the square as uncovered.
                board[state] = 0.5
                uncovered_cell = state
        elif action == 1:  # Mark square as a mine
            reward = 1  # Give a smaller positive reward for marking a square as a mine.
            # Mark the square as a mine.
            board[state] = -0.5
            uncovered_cell = None
            reward = 1
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
        moves += 1

    return total_reward

def train_agent(agent, num_episodes):
    rewards = np.array([])
    epsilons = np.array([])

    for episode in range(num_episodes):
        episode_reward = play_game(agent, build_board(agent.board_size), agent.board_size) 
        rewards = np.append(rewards, episode_reward)
        epsilons = np.append(epsilons, agent.epsilon)  
        agent.update_epsilon()  
        
    mean_value = np.mean(rewards)
    median_value = np.median(rewards)
    stddev_value = np.std(rewards)

    print("Episodes: ", num_episodes)
    print("Mean: ", mean_value)
    print("Median: ", median_value)
    print("Standard Deviation: ", stddev_value)

    wins = 0
    for reward in rewards:
        if reward > agent.board_size ** 2 - 1:
            wins += 1

    x = range(num_episodes)

    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 4, 1)
    plt.plot(x, rewards, label='Total Reward')
    plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean Reward')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Total rewards over all episodes in training')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.hist(rewards, bins=agent.board_size, color='skyblue', edgecolor='black')
    plt.title('Histogram of Episode Rewards')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')

    plt.subplot(2, 4, 3)
    plt.boxplot(rewards)
    plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean Reward')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Total rewards over all episodes in training')
    plt.legend()

    print(wins)
    plt.subplot(2, 4, 4)
    labels = ['Wins', 'Losses']
    sizes = [wins, num_episodes - wins]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode the 'Wins' slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Wins vs. Losses (Board Size: {})'.format(agent.board_size))

    ''' Plot epsilon decay
    plt.subplot(2, 5, 3)
    plt.plot(x, epsilons, label='Epsilon', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay over all episodes in training')
    plt.legend()
    '''

    plt.show()

def main():
    # Edit these values to adjust experiment
    board_size = 5
    number_of_episodes = 100

    agent = QLearningAgent(board_size)
    train_agent(agent, number_of_episodes)

if __name__ == "__main__":
    main()