# Define memory for Experience Replay
from collections import deque
import random

## To sum up:
# 1) deque is a memory structure that has a finite size. When a deque is full and you still store (append) new data into it, the last data stored in the deque will be purged out (removede) while the new data will be added to the first element of deque, and vice versa (In other words, deque practices first in first out [FIFO]).
# 2) Here, we use deque to store the experience to train the DQN. The experience (similar to a set of features for neural network training) here is called transition. Transition is a variable stores the information of (state, action, new_state, reward, terminated).
# 3) state -> the 12 state information of current state
# 4) action -> the action (represented by either 0 or 1) that took by the agent to move into the next state from the current state
# 5) new_state -> the 12 state information of next state
# 6) reward -> the amount of reward the agent received after taking that action to move into the next state from the current state
# 7) terminated -> if the game ended (or is the agent dead in the environment) after the agent taking that action to move into the next state from the current state


class ReplayMemory():

    # maxlen refers to the maximum length used to initialize the deque. seed is used to control the randomness (we can send in a seed to initialize random)
    def __init__(self, maxlen, seed = None):
        self.memory = deque([], maxlen = maxlen)

        # Optional seed for reproducibility 
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition) # append() is used to append the experience to the memory (deque). The experience (features) here is called transition, it is a tuple of information consisting (state, action, new_state, reward, terminated), is used to train the DQN. 

    def sample(self, sample_size): # sample() will randomly select the samples in the transition (experience) then return them. The number of samples that randomly selected depends on the batch size we specified as sample_size.
        return random.sample(self.memory, sample_size) 
    
    def __len__(self): # __len__() will just return the length of the memory (deque)
        return len(self.memory)