import random
import gym
import numpy as np

# Create and reset the environment
env = gym.make("MountainCar-v0")
env.metadata["render_fps"] = 100000
state, _ = env.reset()

# Q-learning parameters
c_learning_rate = 0.1
c_discount = 0.99  # Increased discount factor for long-term planning
episodes = 3000

steps = []
# Q-table parameters
q_table_size = [20, 20]
q_table_segment_size = (
    (env.observation_space.high) - (env.observation_space.low)
) / q_table_size


# Initialize Q-table
def convert_state(state):
    q_state = (state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(np.int64))


q_table = np.random.uniform(
    low=-0.1, high=0, size=(q_table_size + [env.action_space.n])
)

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    i = 0

    while not done:
        action = np.argmax(q_table[convert_state(state)])
        next_real_state, reward, done, _, _ = env.step(action)

        next_state = convert_state(next_real_state)
        if not done:
            # Update Q
            current_q_value = q_table[convert_state(state) + (action,)]
            next_action = np.argmax(q_table[next_state])
            next_q_value = q_table[next_state + (next_action,)]
            new_q_value = current_q_value + c_learning_rate * (
                reward + c_discount * next_q_value - current_q_value  # next state value
            )
            q_table[convert_state(state) + (action,)] = new_q_value
            state = next_real_state
            i += 1
        else:
            if next_real_state[0] > env.goal_position:
                # print(f"Episode {episode} - hit goal in {i} steps")
                steps.append(i)
            else:
                print(f"Episode {episode} - miss goal")

    # Render only at the end of specific episodes

print(np.median(steps))
print(np.min(steps))

env.close()
