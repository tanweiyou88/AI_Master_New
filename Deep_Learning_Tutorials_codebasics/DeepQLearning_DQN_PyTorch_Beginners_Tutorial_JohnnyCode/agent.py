"""
# Implement Deep Q-Network Module (DQN PyTorch Beginners Tutorial 2, by Johny Code)

Link to the Youtube video tutorial: https://www.youtube.com/watch?v=RVMpm86equc&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=2
Link to the Gymnasium official website: https://gymnasium.farama.org/ 
Link to the Flappy Bird environment GitHub repository: https://github.com/markub3327/flappy-bird-gymnasium 

**Information of this tutorial:**
1) agent.py is the main script, dqn.py is the supplementary script
2) In dqn.py, we define the architecture of the DQN. While in agent.py, the script is same except we import the DQN from dpn.py into agent.py to train this DQN based on the current state features so that the trained DQN has the policy that provides the best action for the agent to take to move into next state, given the features of the current state.
"""
import torch # import pytorch library
import flappy_bird_gymnasium # Import the flappy bird environment, which is compatible with gymnasium
import gymnasium # Import the gymnasium, which is an API standard for reinforcement learning with a diverse collection of reference environments
from dqn import DQN # from the dqn.py located in the same folder, import the DQN class function defined in the dqn.py
from experience_replay import ReplayMemory # from the experience_replay.py located in the same folder, import the ReplayMemory class function defined in the experience_replay.py
import itertools
import yaml # so that we can use YAML file to update the hyperparameters in this main file

# Set the device as 'cuda' to use GPU for processing when a GPU is recognized. Else, set the device as 'cpu' to use CPU for processing.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define a class for agent, called Agent
class Agent:
    
    def __init__(self, hyperparameter_set): # we pass the hyperparameter set name we want (the argument to replace hyperparameter_set) to this __init__ function.
        with open('hyperparameters.yml', 'r') as file: # open the file called 'hyperparameters.yml', then read the content inside it. This operation is handled by "file".
            all_hyperparameters_sets = yaml.safe_load(file) # Load all the contents (hyperparameter values) in the file called 'hyperparameters.yml' to the variable called all_hyperparameters_sets.
            hyperparameters = all_hyperparameters_sets(hyperparameter_set) # Then, we only store (choose) the hyperparameter set with the name of hyperparameter_set (means store a bunch of hyperparameters which are indented under the hyperparameter set name) to the variable called hyperparameter_set.
            # Here, the only hyperparameter set name we have in the file called 'hyperparameters.yml' is called cartpole1. Under the hyperparameter set cartpole1, we have multiple hyperparameter values (such as env_id, replay_memory_size, ...).
        
        # Hyperparameters (adjustable). The codes below update the hyperparameter variables in this main file using the corresponding hyperparameter values stored in the variable hyperparameters (the hyperparameter values loaded from the YAML file) respectively. 
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
    
    # Define a run function to perform both training and testing. "is_training = True" refers to perform training; "is_training = False" refers to perform testing. "render = False" refers to render the environment (show the environment in a figure window); "render = False" refers to don't render the environment (don't show the environment in a figure window).
    def run(self, is_training = True, render = False):
        
        # Create an instance of the flappy bird environment called env, using the "FlappyBird-v0" environment model and with the help of gymnasium. The parameter "render_mode="human"" is used to render the game on the screen. The parameter "use_lidar=False" is the custom parameter that allow the user to turn on (use the information of) the LIDAR sensor or off (don't use the information of LIDAR).
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False) # This means env is the environment that we create here.
        # env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space[0] # get the number of state information at a state from the observation space of the environment
        num_actions = env.action_space.n # get the number of actions from the action space of the environment

        rewards_per_episode = [] # create a variable to keep check of the amount of reward the agent received per episode (1 episode means the DQN training is executed such that the agent takes actions in the environent to get state information until the training is terminated [terminated = True])
        # In each episode, it can have multiple iterations. Each episode starts with 1st iteration and ends with terminated = True.
        # In each iteration, the agent takes an action to move into the next state from the current state. Also, the state information of the current state and the next state will be obtained in each iteration.

        # create a deep Q-network (also known as policy network) called policy_dqn, by using the DQN function defined in dqn.py. The available device will be used to create the layer's parameters of the DQN.
        policy_dqn = DQN(num_states, num_actions).to_device(device)

        if is_training: # if we are training the DQN
            memory = ReplayMemory(self.replay_memory_size) # we create the memory (deque) of size replay_memory_size. Use the __init__() in experience_replay.py (maybe  __init__() is used when we called the class function in a py file for the 1st time)

        # This for loop is used to train the DQN for infinite iterations/episodes, and manually stops the training when we are satisfied with the results.
        for episode in itertools.count(): # itertools() is a python module that generates number infinitely, 1 number is generated a time (starting from 0, then incremented to 1, 2, 3,... at each time, and vice versa)
            # At the beginning of the episode/iteration (means before the training is started),
            state, _ = env.reset() # call the reset() to reset the environment
            terminated = False # Initialized the terminated variable as False (so the coming while loop will be executed to perform the DQN training)
            episode_reward = 0.0 # Create a variable to store the accumulate reward the agent received throughout the DQN training. So we can count the amount of reward the agent received in current epsisode (from the 1st iteration of current episode [the agent takes actions to move in the environment] until the last iteration of the current episode [the agent is dead in the environment, such that terminated = True])

            while not terminated: # This is an infinite loop for the agent(flappy bird) to interact with the environment through action, observation, and reward.
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample() # use sample() on action space(action_space) of the environment(env) to get a random action for the agent (flappy bird) at each iteration. This means the sample() might return different actions(values) at each iteration. 
                # More explanation: 
                # 1) The action_space of an environment refers to all possible actions that can be taken by the agent. In this environment, the action_space consists of only 2 values(0:the agent[flappy bird] do nothing; 1:the agent[flappy bird] flaps its wing to fly up). 
                # 2) Hence, the sample() here will only return either 0 or 1. While the returned value is stored in the action variable.


                # Processing:
                new_state, reward, terminated, _, info = env.step(action) # The action selected is passed into the step() to execute that action.
                # More explanation:
                # 1) The step() will give us back the state information at the next state (new_state) [means what are the 12 state information observed by the agent at the next state in the form of values, after the agent took that action to move into the next state from the current state], reward [means how much reward we got from the last action],
                # terminated [terminated=True -> if the bird hits the ground or one of the pipes; else, terminated=False], _ means the parameter is not used,
                # info [just contains the additional information you can use for debugging or something]
                # 2) new_state
                #   2.1) The new_state is a vector that contains 12 elements, while each element represents a state information as a value (according to the repository, there are 12 different state information considered as observation). The 12 elements represent different state information respectively in sequence, according to the repository of the environemnt. Hence, the first element of the obs vector represents the state information of the last pipe's horizontal position,...., the last element of the obs vector represents the state information of the player's rotation.
                #   2.2) The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1 (because the author used this range to train the neural network)

                # Accumulate reward
                episode_reward += reward # Add the amount of reward the agent received in this iteration to the amount of reward the agent received previously.

                if is_training: # if we are training the DQN
                    memory.append((state, action, new_state, reward, terminated)) # Use the append() in experience_replay.py. "(state, action, new_state, reward, terminated)" here equals to the "transition" in experience_replay.py.

                # At here, the agent already took the action to move into the next state from the current state. So, before entering the next iteration, we update the next_state (variable in this script) to the state (variable in this script) to replace the old information in state (variable in this script).
                state = new_state

            rewards_per_episode.append(episode_reward)
        

# **The information obtained at the first iteration of the loop, through debug console (refers to D:\AI_Master_New\Under_Local_Git_Covered\Deep_Learning_Tutorials_codebasics\DeepQLearning(DQN)_PyTorch_Beginners_Tutorial_JohnnyCode\Implement_Deep_Q_Learning_with_PyTorch_and_Train_Flappy_Bird_DQN_PyTorch_Beginners_Tutorial1\Explain (don't run)\hidden\photo1.png):**
# 1) The action selected is 0, means the flappy bird do nothing.
# 2) After the action is executed, the observation of 12 different state information is stored in obs vector as image above. The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1.
# 3) The env.observation_space shows that each element value is normalized to the range between -1 and 1, while the size of the observation space is 12 (that's why the obs vector consists of 12 elements). Observation space refers to all the state information gathered in an iteration (when the agent arrives at next state after executing the action in existing iteration/state)
# 4) The reward the agent gained when it arrives at the next state after executing the action in existing iteration/state is 0.1. According to the Flappy Bird environment GitHub repository, the reward of 0.1 means the agent (flappy bird) stays alive when it arrives at the next state.