import retro
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
from collections import deque
from skimage import transform, color
import matplotlib.pyplot as plt 

env=retro.make(game='SpaceInvaders-Atari2600')
# Box(210,160,3)

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

def preprocess_frame(frame):
     # Greyscale frame 
     gray = color.rgb2gray(frame)
     # Crop the screen (remove the part below the player)
     # [Up: Down, Left: right]
     cropped_frame = gray[8:-12,4:-12]
     # Normalize Pixel Values
     normalized_frame = cropped_frame/255.0 
     # Resize
     preprocessed_frame = transform.resize(normalized_frame, [110,84])
     return preprocessed_frame 
 
stack_size = 4 # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
def stack_frames(stacked_frames, state, is_new_episode):
     # Preprocess frame
     frame = preprocess_frame(state) 
     if is_new_episode:
         # Clear our stacked_frames
         stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 
         # Because we're in a new episode, copy the same frame 4x
         stacked_frames.append(frame)
         stacked_frames.append(frame)
         stacked_frames.append(frame)
         stacked_frames.append(frame)
         
         # Stack the frames
         stacked_state = np.stack(stacked_frames, axis=2) 
     else:
         # Append frame to deque, automatically removes the oldest frame
         stacked_frames.append(frame)
         # Build the stacked state (first dimension specifies different frames)
         stacked_state = np.stack(stacked_frames, axis=2) 
     return stacked_state, stacked_frames
 
    
### MODEL HYPERPARAMETERS
state_size = [110, 84, 4] # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
action_size = env.action_space.n # 8 possible actions
learning_rate = 0.00025 # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 200 # Total episodes for training
max_steps = 50000 # Max possible steps in an episode
batch_size = 64 # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0 # exploration probability at start
explore_stop = 0.01 # minimum exploration probability 
decay_rate = 0.00001 # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9 # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000 # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

memory = deque(maxlen=100000)

def sample(memory, batch_size):
     buffer_size = len(memory)
     index = np.random.choice(np.arange(buffer_size),
     size = batch_size,
     replace = False) 
     return [memory[i] for i in index]
 
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.append((state, action, reward, next_state, done))
        
        # Start a new episode
        state = env.reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        memory.append((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
        state = next_state
    
def predict_action(model,explore_start, explore_stop, decay_rate, decay_step, state, actions):
     exp_exp_tradeoff = np.random.rand()
     explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
     if (explore_probability > exp_exp_tradeoff):
         choice = random.randint(1,len(possible_actions))-1
         action = possible_actions[choice]
     else:
         Qs = model.predict(state.reshape((1, *state.shape)))
         choice = np.argmax(Qs)
         action = possible_actions[choice]
     return action, explore_probability


def replay(agent,batch_size,memory):
     minibatch = sample(memory,batch_size)
     for state, action, reward, next_state, done in minibatch:
         target = reward
         if not done:
             target = reward + gamma*np.max(agent.predict(next_state.reshape((1,*next_state.shape)))[0])
             target_f = agent.predict(state.reshape((1,*state.shape)))
             target_f[0][action] = target
             agent.fit(state.reshape((1,*state.shape)), target_f, epochs=1, verbose=0)
     return agent    
    
def DQNetwork():
     model=Sequential()
     model.add(Convolution2D(16,input_shape=(110,84,4),kernel_size=8, strides=4, padding='valid',activation='elu'))
     model.add(Convolution2D(32, kernel_size=4, strides=2, padding='valid',activation='elu'))
     model.add(Convolution2D(64, kernel_size=3, strides=2, padding='valid',activation='elu'))
     model.add(Flatten())
     model.add(Dense(units=512))
     model.add(Dense(units=8,activation='softmax'))
     model.compile(optimizer=Adam(0.01),loss='mse')
     return model
 
    
agent = DQNetwork()
#Loading the latest model
#agent = keras.models.load_model("Training episode 299_.h5")     
agent.summary()
rewards_list=[]
Episodes = 10

# Training learning measurement variables
current_step = 0
step_number = []
accumulated_reward = []
accumulated_avg = []
all_rewards = 0
rewards = []
episodes = []

# Iterate the game
for episode in range(Episodes):
     # reset state in the beginning of each game
     step = 0
     decay_step = 0
     avg_rewards = 0
     total_rewards = 0
     episode_rewards = []
     state = env.reset()
     state, stacked_frames = stack_frames(stacked_frames, state, True)     
     
     while step < max_steps:
         step += 1
         decay_step +=1
# Predict the action to take and take it
         action, explore_probability = predict_action(agent,explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
 #Perform the action and get the next_state, reward, and done information
         next_state, reward, done, _ = env.step(action)
         # Add the reward to total reward
         episode_rewards.append(reward)
 # Append learning measurement data
         all_rewards += reward
         avg_rewards += reward
         total_rewards += reward
         accumulated_reward.append(all_rewards)
         accumulated_avg.append(avg_rewards)
         step_number.append(current_step)
         # Increment current step number for learning measurement graph
         current_step += .001
     if done:
 # The episode ends so no next state
         next_state = np.zeros((110,84), dtype=np.int)
         next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
 # Set step = max_steps to end the episode
         step = max_steps
 # Get the total reward of the episode
         total_reward = np.sum(episode_rewards)
         print('Episode:{}/{} Score:{} Explore Prob:{}'.format(episode,Episodes,total_reward,explore_probability))
         rewards_list.append((episode, total_reward))
         rewards.append(total_rewards)
         episodes.append(episode)
 # Store transition <st,at,rt+1,st+1> in memory D
         memory.append((state, action, reward, next_state, done))
     else:
 # Stack the frame of the next_state
         next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
 # Add experience to memory
         memory.append((state, action, reward, next_state, done))
 # st+1 is now our current state
         state = next_state
     env.render() 
 # train the agent with the experience of the episode
     agent=replay(agent,batch_size,memory)
     agent.save("Training episode "+str(episode)+'_.h5')
     agent = keras.models.load_model("Training episode "+str(episode)+'_.h5')

# Training - Total Reward vs Step Number graph
plt.plot(step_number, accumulated_reward, linewidth=1.0)
plt.title('Total Reward vs Step Number for DQN Training')
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
     
score=[]
episode=[]
for e,r in rewards_list:
     episode.append(e)
     score.append(r)
plt.plot(episode,score)
plt.show()




#======LIVE GAMEPLAY AFTER TRAINING========
    
# reset state in the beginning of each game
total_test_episodes = 5

accumulated_reward = []
current_step = 0
step_number = []
rewards = []
episodes = []
for episode in range(total_test_episodes):
     step = 0
     all_rewards = 0
     total_rewards = 0
     decay_step = 0
     explore_start = .1
     episode_rewards = []
     state = env.reset()
     state, stacked_frames = stack_frames(stacked_frames, state, True)
     done = False
     #agent_loaded = keras.models.load_model('Training episode 184_.h5')
     while not done:
         step += 1
         decay_step +=1
         # Predict the action to take and take it
         action, explore_probability = predict_action(agent,explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
     #Perform the action and get the next_state, reward, and done information
         next_state, reward, done, _ = env.step(action)
         # Add the reward to total reward
         episode_rewards.append(reward)
         # Append learning measurement data
         all_rewards += reward
         total_rewards += reward
         accumulated_reward.append(all_rewards)
         step_number.append(current_step)
         # Increment current step number for learning measurement graph
         current_step += .001
         if done:
              rewards.append(total_rewards)
              episodes.append(episode)
         env.render()
         
     total_reward = np.sum(episode_rewards)
     print('Score:{}'.format(total_reward))

     env.render

env.close()
print ("Average Score: ",  (all_rewards)/total_test_episodes)
# Test - Total Reward vs Step Number graph
plt.plot(step_number, accumulated_reward, linewidth=1.0)
plt.title('Reward per Game vs Step Number for DQN Test')
plt.ylabel('Accumulated Reward')
plt.xlabel('Numbers of Steps (in thousands)')
plt.show()
    
plt.bar(episodes, rewards, linewidth=1.0)
plt.title('Reward per Game for DQN Test')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
