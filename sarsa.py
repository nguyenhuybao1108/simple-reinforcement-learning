import numpy as np
import gym

# Create and reset the environment
env = gym.make("MountainCar-v0")
env.metadata["render_fps"] = 100000
state, _ = env.reset()

# SARSA parameters
c_learning_rate = 0.1
c_discount = 0.99  # Discount factor for long-term planning
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 3000

steps = []
# Q-table parameters
q_table_size = [20, 20]
q_table_segment_size = (
    env.observation_space.high - env.observation_space.low
) / q_table_size


# Initialize Q-table
def convert_state(state):
    q_state = (state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(np.int64))


q_table = np.random.uniform(
    low=-0.1, high=0, size=(q_table_size + [env.action_space.n])
)


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.choice(env.action_space.n)
    else:
        # Exploit: choose the action with the highest Q-value
        return np.argmax(q_table[state])


for episode in range(episodes):
    state, _ = env.reset()
    state = convert_state(state)
    action = choose_action(state, epsilon)
    done = False
    i = 0

    while not done:
        next_real_state, reward, done, _, _ = env.step(action)
        next_state = convert_state(next_real_state)
        next_action = choose_action(next_state, epsilon)

        # Update Q
        current_q_value = q_table[state + (action,)]
        next_q_value = q_table[next_state + (next_action,)]
        new_q_value = current_q_value + c_learning_rate * (
            reward + c_discount * next_q_value - current_q_value
        )
        q_table[state + (action,)] = new_q_value

        state = next_state
        action = next_action
        i += 1

    # Record steps
    if next_real_state[0] > env.goal_position:
        steps.append(i)
    else:
        print(f"Episode {episode} - miss goal")

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Optionally render the environment at specific episodes

print(f"Median steps: {np.median(steps)}")
print(f"Minimum steps: {np.min(steps)}")

env.close()
