import numpy as np
import gym
import random
import matplotlib.pyplot as plt 

#env = gym.make("Taxi-v3")
env = gym.make("SpaceInvaders-v0")
env.render()


action_size = env.action_space.n
print("Action size ", action_size)

#state_size = env.observation_space
state_size = 160
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))


total_episodes = 200            # Total episodes
max_steps = 10000               # Max steps per episode

learning_rate = 0.3             # Learning rate
gamma = 0.7                     # Discounting rate

# Exploration parameters
epsilon = 1.0                   # Exploration rate
max_epsilon = 1.0               # Exploration probability at start
min_epsilon = 0.01              # Minimum exploration probability 
decay_rate = 0.001              # Exponential decay rate for exploration prob



# Training learning measurement variables
current_step = 0
step_number = []
accumulated_reward = []
accumulated_avg = []
all_rewards = 0
rewards = []
episodes = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):   
    print('Episode ', episode+1,'/',total_episodes)
    # Reset the environment
    state = env.reset()
    # Initialize state to zero
    state = 0
    step = 0
    avg_rewards = 0
    total_rewards = 0
    done = False
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state_array, reward, done, info = env.step(action)
 
        # Append learning measurement data
        all_rewards += reward
        avg_rewards += reward
        total_rewards += reward
        accumulated_reward.append(all_rewards)
        accumulated_avg.append(avg_rewards)
        step_number.append(current_step)
        # Increment current step number for learning measurement graph
        current_step += .001
        
        # Finding the new location of the new state
        new_state = 0
        for i in range(159):
            if new_state_array[185][i][0] == 50:
                new_state = i
                break
           
            
            
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        
        # If done : finish episode
        if done == True:
            rewards.append(total_rewards)
            episodes.append(episode)
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

# Save the Q-table with the training info
#np.save('Qtable', qtable)

# Training - Total Reward vs Step Number graph
plt.plot(step_number, accumulated_reward, linewidth=1.0)
plt.title('Total Reward vs Step Number for Q-Learning Training')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()

# Training - Total Reward vs Step Number graph
plt.plot(step_number, accumulated_avg, linewidth=1.0)
plt.title('Single Game Score vs Step Number for Q-Learning Training')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()

plt.bar(episodes, rewards, linewidth=1.0)
plt.title('Reward per Game for Q-Learning Training')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
