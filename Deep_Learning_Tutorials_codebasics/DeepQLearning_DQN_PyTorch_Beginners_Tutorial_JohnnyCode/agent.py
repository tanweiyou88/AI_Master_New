"""
# Implement Deep Q-Network Module (DQN PyTorch Beginners Tutorial 2, by Johny Code)

Link to the Youtube video tutorial: https://www.youtube.com/watch?v=RVMpm86equc&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=2
Link to the Gymnasium official website: https://gymnasium.farama.org/ 
Link to the Flappy Bird environment GitHub repository: https://github.com/markub3327/flappy-bird-gymnasium 

**Information of this tutorial:**
1) agent.py is the main script, dqn.py is the supplementary script
2) In dqn.py, we define the architecture of the DQN. While in agent.py, the script is same except we import the DQN from dpn.py into agent.py to train this DQN based on the current state features so that the trained DQN has the policy that provides the best action for the agent to take to move into next state, given the features of the current state.
"""
import torch # import PyTorch library
from torch import nn # import nn module from PyTorch library so we can use nn.
import flappy_bird_gymnasium # Import the flappy bird environment, which is compatible with gymnasium
import gymnasium as gym # Import the gymnasium, which is an API standard for reinforcement learning with a diverse collection of reference environments
import numpy as np
import yaml # so that we can use YAML file to update the hyperparameters in this main file
import random # so that we can use random() to generate a random floating number between 0 and 1.0
import matplotlib
import matplotlib.pyplot as plt
import os 
from datetime import datetime, timedelta
import argparse
import itertools

from dqn import DQN # from the dqn.py located in the same folder, import the DQN class function defined in the dqn.py
from experience_replay import ReplayMemory # from the experience_replay.py located in the same folder, import the ReplayMemory class function defined in the experience_replay.py


##--------------------IMPORTANT NOTES--------------------
# When we are using PyTorch to develop DQN, we need to make sure the things that are going into the DQN are tensor object. In this tutorial, the things that are going into the DQN are the components of experience (state, action, new_state, reward, terminated).
##-------------------------------------------------------

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info (create a folder called runs that stores the information of log, graphs, trained model,...)
RUNS_DIR = "D:/AI_Master_New/Under_Local_Git_Covered/Deep_Learning_Tutorials_codebasics/DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode/runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg' : used to generate plots as images and save them to a file, instead of rendering to screen
matplotlib.use('Agg')

# Set the device as 'cuda' to use GPU for processing when a GPU is recognized. Else, set the device as 'cpu' to use CPU for processing.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # force to use CPU

# Define a class for agent, called Agent
class Agent:
    
    def __init__(self, hyperparameter_set): # we pass the hyperparameter set name we want (the argument to replace hyperparameter_set) to this __init__ function.
        with open('Deep_Learning_Tutorials_codebasics\DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode\hyperparameters.yml', 'r') as file: # open the file called 'hyperparameters.yml', then read the content inside it. This operation is handled by "file".
            all_hyperparameters_sets = yaml.safe_load(file) # Load all the contents (hyperparameter values) in the file called 'hyperparameters.yml' to the variable called all_hyperparameters_sets.
            hyperparameters = all_hyperparameters_sets[hyperparameter_set] # Then, we only store (choose) the hyperparameter set with the name of hyperparameter_set (means store a bunch of hyperparameters which are indented under the hyperparameter set name) to the variable called hyperparameter_set.
            # Here, the only hyperparameter set name we have in the file called 'hyperparameters.yml' is called cartpole1. Under the hyperparameter set cartpole1, we have multiple hyperparameter values (such as env_id, replay_memory_size, ...).
        
        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable). The codes below update the hyperparameter variables in this main file using the corresponding hyperparameter values stored in the variable hyperparameters (the hyperparameter values loaded from the YAML file) respectively. 
        self.env_id             = hyperparameters['env_id']                 # the environment model
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount factor (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']              # the number of neurons (nodes) in the hidden layer called fc1 of the neural network
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        # Hyperparameters for the neural network
        self.loss_fn = nn.MSELoss() # Neural network loss function. MSE = Mean Squared Error, can be swapped to other types of loss function.
        self.optimizer = None   # Neural network optimizer. Initialize later.

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log') # store the log file at the location of RUN_DIR, with the filename called hyperparameter_set
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt') # store the trained model at the location of RUN_DIR, with the filename called hyperparameter_set
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png') # store the graph at the location of RUN_DIR, with the filename called hyperparameter_set

    # Define a run function to perform both training and testing. "is_training = True" refers to perform training; "is_training = False" refers to perform testing. "render = False" refers to render the environment (show the environment in a figure window); "render = False" refers to don't render the environment (don't show the environment in a figure window).
    def run(self, is_training = True, render = False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        '''
        # Create an instance of the flappy bird environment called env, using the "FlappyBird-v0" environment model and with the help of gymnasium. The parameter "render_mode="human"" is used to render the game on the screen. The parameter "use_lidar=False" is the custom parameter that allow the user to turn on (use the information of) the LIDAR sensor or off (don't use the information of LIDAR).
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False) # This means env is the environment that we create here.
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        '''
        
        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Get observation space size
        num_states = env.observation_space.shape[0] # get the number of state information at a state from the observation space of the environment 
        
        # Number of possible actions
        num_actions = env.action_space.n # get the number of actions from the action space of the environment 

        # List to keep track of rewards collected per episode.
        rewards_per_episode = [] # create a variable to keep check of the amount of reward the agent received per episode (1 episode means the DQN training is executed such that the agent takes actions in the environent to get state information until the training is terminated [terminated = True])
        # In each episode, it can have multiple iterations. Each episode starts with 1st iteration and ends with terminated = True.
        # In each iteration, the agent takes an action to move into the next state from the current state. Also, the state information of the current state and the next state will be obtained in each iteration.
        
        # Create a deep Q-network (also known as policy network) called policy_dqn, by using the DQN function defined in dqn.py. The available device will be used to create the layer's parameters of the DQN.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training: # if we are training the DQN
            
            # Initialize epsilon
            epsilon = self.epsilon_init # initialize the epsilon with value of 1 (because epsilon_init = 1 in hyperparamter.yml), the parameter for the Epsilon Greedy algorithm

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size) # we create the memory (deque) of size replay_memory_size. Use the __init__() in experience_replay.py (maybe  __init__() is used when we called the class function in a py file for the 1st time)
            
            # Create a target network called target_dqn and make it identical to the policy network (DQN), by using the DQN function defined in dqn.py. The available device will be used to create the layer's parameters of the DQN.
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            # Load (copy) all the weights and bias of the DQN to the target network, similar to transfer learning. So now, the target network has the same weights and bias of the DQN (both DQN and the target network are synced, in terms of weights and bias).
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to other types of optimizer. We provide the DQN paramters to the optimizer so that the optimizer knows how to optimize the policy.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network
            step_count = 0

            # Use the variable best_reward to track the best reward. So initialize the variable best_reward with a very small number first (EG:-9999999)
            best_reward = -9999999

        else:  # if we are NOT training the DQN
            
            # Load the learned policy (the trained DQN model) by loading the parameters (weights and biases) of the trained model from the MODEL_FILE
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Switch the model to the evaluation mode
            policy_dqn.eval()

        # This for loop is used to train the DQN for infinite iterations/episodes, and manually stops the training when we are satisfied with the results.
        for episode in itertools.count(): # itertools() is a python module that generates number infinitely, 1 number is generated a time (starting from 0, then incremented to 1, 2, 3,... at each time, and vice versa)
            # At the beginning of the episode/iteration (means before the training is started),
            state, _ = env.reset() # call the reset() to reset the environment
            
            # Convert the variable state into tensor object
            state = torch.tensor(state, dtype=torch.float, device=device) # Since we are using PyTorch to develop DQN, we need to make sure the things that are going into the DQN are tensor object. In this tutorial, the things that are going into the DQN are the components of experience (state, action, new_state, reward, terminated). Since the variable state is the thing that goes into the DQN, we need to convert it into a tensor object, using the tensor function. After we passing the variable state into the tensor function, the variable state is converted into tensor object and now its values become floating number (because dtype=torch.float). Then, we will send the converted variable state to the device we selected (either CPU or GPU) for processing. 

            terminated = False # Initialized the terminated variable as False (so the coming while loop will be executed to perform the DQN training)
            episode_reward = 0.0 # Create a variable to store the accumulate reward the agent received throughout the DQN training. So we can count the amount of reward the agent received in current epsisode (from the 1st iteration of current episode [the agent takes actions to move in the environment] until the last iteration of the current episode [the agent is dead in the environment, such that terminated = True])

            # Perform actions until episode terminates or reaches the amount of reward you specified
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop_on_reward is necessary)
            while (not terminated and episode_reward < self.stop_on_reward): # This is an infinite loop for the agent(flappy bird) to interact with the environment through action, observation, and reward.
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon: # If we are training the DQN and the random number generated by "random.random()" is less than epsilon, we will do a random action (means the agent will take a random action). When we just start the DQN training and epsilon=1, most likely we will enter this section (because the number generated by random() is between 0 to 1)
                    action = env.action_space.sample() # use sample() on action space(action_space) of the environment(env) to get a random action for the agent (flappy bird) at each iteration. This means the sample() might return different actions(values) at each iteration. 
                    # More explanation: 
                    # 1) The action_space of an environment refers to all possible actions that can be taken by the agent. In this environment, the action_space consists of only 2 values(0:the agent[flappy bird] do nothing; 1:the agent[flappy bird] flaps its wing to fly up). 
                    # 2) Hence, the sample() here will only return either 0 or 1. While the returned value is stored in the action variable.
                    
                    # Convert the variable state into tensor object
                    action = torch.tensor(action, dtype=torch.int64, device=device) # Since we are using PyTorch to develop DQN, we need to make sure the things that are going into the DQN are tensor object. In this tutorial, the things that are going into the DQN are the components of experience (state, action, new_state, reward, terminated). Since the variable action is the thing that goes into the DQN, we need to convert it into a tensor object, using the tensor function. After we passing the variable action into the tensor function, the variable action is converted into tensor object and now its values become 64-bit signed interger number (because dtype=torch.int64). Then, we will send the converted variable state to the device we selected (either CPU or GPU) for processing. 

                else: # otherwise, we will select the action that the DQN (policy network) prescribes
                    with torch.no_grad(): # here we are estimating for the best action. PyTorch does gradient calculation automatically during the DQN training. Since at here, we're not doing training, we're just evaluating a state (choosing the best action provided by the DQN), so we can turn off the gradient calculation of PyTorch using this line, just to save on processing power.
                        # we provide "state" as the input to the policy_dqn (the DQN), then policy_dqn will provide the Q values of all actions (In this environment, since there are only 2 actions in the action space, the DQN will only provide 2 Q values as the output). 
                        # Since we want the agent to take the action with the highest Q value & the index of Q values returned by the DQN represents an action [here, index 0 represents the agent do nothing; index 1 represents the agent flaps its wing], ".argmax()" is used to return the index of the highest Q value. 
                        # Hence, the index of the highest Q value is selected as the action prescribes by the DQN for the agent to take to move into next state from the current state.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() 
                        # "unsqueeze(dim=0)" is appied to variable state to add 1 more dimension at its very front (so now 1-dimensional tensor([1,2,3,...]) becomes 2-dimensional tensor([[1,2,3,...]])). The newly added dimension becomes the 1st dimension of the variable state that stores the information of batch.
                        # Since the input of the DQN, variable state now has become 2-dimensional, the output of the DQN also becomes 2-dimensional. The output of the DQN are the Q values of all actions in the action space. Hence, we apply "squeeze()" on the DQN outputs to squeenze the outputs into 1-dimensional, so now we can get the index of each Q value provided by the DQN.
                        

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item()) # The action selected is passed into the step() to execute that action.
                # Since the input of the DQN, variable state is a tensor object, the output of the DQN is also a tensor object. The output of the DQN are the Q values of all actions in the action space. Hence, we apply "item()" on the variable action to get the value of the variable action (which is a tensor object).

                # More explanation:
                # 1) The step() will give us back the state information at the next state (new_state) [means what are the 12 state information observed by the agent at the next state in the form of values, after the agent took that action to move into the next state from the current state], reward [means how much reward we got from the last action],
                # terminated [terminated=True -> if the bird hits the ground or one of the pipes; else, terminated=False], _ means the parameter is not used,
                # info [just contains the additional information you can use for debugging or something]
                # 2) new_state
                #   2.1) The new_state is a vector that contains 12 elements, while each element represents a state information as a value (according to the repository, there are 12 different state information considered as observation). The 12 elements represent different state information respectively in sequence, according to the repository of the environemnt. Hence, the first element of the obs vector represents the state information of the last pipe's horizontal position,...., the last element of the obs vector represents the state information of the player's rotation.
                #   2.2) The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1 (because the author used this range to train the neural network)

                # Accumulate reward
                episode_reward += reward # Add the amount of reward the agent received in this iteration to the amount of reward the agent received previously.

                # Convert the variable new_state and reward into tensor object. Since we are using PyTorch to develop DQN, we need to make sure the things that are going into the DQN are tensor object. In this tutorial, the things that are going into the DQN are the components of experience (state, action, new_state, reward, terminated). Since the variable new_state and reward are the thing that goes into the DQN, we need to convert each of them into tensor objects respectively, using the tensor function. After we passing the variable new_state and reward into the tensor function respectively, each of them is converted into tensor object and now its values become floating number (because dtype=torch.float). Then, we will send the converted variable state to the device we selected (either CPU or GPU) for processing. 
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training: # if we are training the DQN
                    memory.append((state, action, new_state, reward, terminated)) # Use the append() in experience_replay.py. "(state, action, new_state, reward, terminated)" here equals to the "transition" in experience_replay.py.

                    # Increment step counter
                    step_count += 1
        

                # At here, the agent already took the action to move into the next state from the current state. So, before entering the next iteration, we update the next_state (variable in this script) to the state (variable in this script) to replace the old information in state (variable in this script).
                state = new_state

            # Keep track of the rewards collected per episode
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained
            if is_training: # if we are training the DQN
                if episode_reward > best_reward: # if the epsiode reward is greater than the best reward
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model.."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE) # save the model (means save the parameters [EG: weights and biases of the neural network] as file at the location of MODEL_FILE)
                    best_reward = episode_reward # Replace the previous best reward with this current best reward
            
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
            
            # Check if enough experience has been collected
            if len(memory)>self.mini_batch_size: # if yes,
                
                # Sample from memory. Take a mini_batch_size numbers of experience instances of the memory and store them into variable mini_batch. 
                mini_batch = memory.sample(self.mini_batch_size)

                # Create a new function called optimize. We provide the optimize function with the sampled experience as input, then the sampled experience will received by both the DQN and target network for syncing.
                self.optimize(mini_batch,policy_dqn, target_dqn)

                ## Decay epsilon
                # Obviously, at the beginning when the DQN is untrained, DQN is going to spit out garbage. But as we train the DQN, the policy gets better (means the DQN provide better outputs), and we'll get better actions from the policy (DQN).
                # So, we want to slowly decrease the epsilon after 1 episode, by performing epsilon multiplying with epsilon_decay (actually there are different methods to decrease the epsilon). We take the maximum value between the decreased epsilon and the minimum epsilon to make sure the epsilon does not go under the minimum.
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                # keep track of the epsilon history at each episode
                epsilon_history.append(epsilon)

                # Copy the weights and biases of DQN to the target network after a certain number of steps (when the step_count is greater than the network_sync_rate)
                if step_count > self.network_sync_rate: # when the certain number of steps is reached
                    target_dqn.load_state_dict(policy_dqn.state_dict()) # load (copy) all the weights and biases of the DQN to the target network, similar to transfer learning. So now, the target network has the same weights and biases of the DQN (both DQN and the target network are synced, in terms of weights and biases).
                    step_count = 0 # reset the step_count back to 0. This means both the DQN and the target network will only be synced (in terms of weights and biases) at every certain steps.

    # Define the save_graph() that generates the graphs of rewards_per_episode and epsilon_history respectively, before saving them to the location of GRAPH_FILE.
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # Define the optimize function that optimizes the policy network (DQN). This optimize function takes the sampled experience to feed the DQN (DQN takes the sampled experience as features to perform training) and feed the target network (so target network can generate target values to improve DQN outputs)
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        ## Important notes: The output of the deep Q-network (DQN) is the predicted Q values for all actions (the predicted reward of taking the corresponding actions) and taget network are the target Q values for all actions (the ground truth reward of taking the corresponding actions) in the action space of the environment. A target network is a separate neural network used in deep reinforcement learning algorithms to stabilize the learning process. It is a copy of the main Q-network, but its parameters are updated less frequently, providing more stable target values for the Q-learning update rule.
        # The purpose of syncing both DQN and target network at every certain count step:
        # 1) At the 1st sync, since we use mini batch here, the DQN is used to provide the predicted Q values for all actions at all instances in the memory (deque). The output of DQN is used get the loss function by comparing it with the outputs of the target network to train the target network. The target network is trained with the loss function & both the DQN and target network have the same weights and biases. At each iteration of different episodes before the next sync, only DQN training is conducted. The DQN will provide outputs with some randomness [because we implement Random Greedy algorithm] (predicted Q-values for taking different actions) based on the features in the memory (deque) it has, and also to collect more experience. But when more and more iteration performed, the data in the memory changed and this caused the DQN to provide different outputs (the predictions are fluactuated, so the DQN training can't be conducted stably in dynamic environment).
        # 2) Hence, we need the trained target network to provide outputs (target Q values for all actions in the action space). Since the target network is synced (its weights and biases) and trained only once for every few step counts, the weigths and biases of the target network is remained constant for a period so that the target network can provide stable target values at that period for DQN training.
        # 3) In other words, the target network outputs (target Q values) are used as the guidance (we now can calculate the loss function using the target Q values from the target network and predicted Q values from the DQN) to realize the backpropagation of DQN for its learning.
        # 4) When approaching to the next sync, due to the backpropagation, the DQN is trained to provide better outputs (the outputs Q values are closer to the one of the target network). This also means the target network guides the DQN in its training.
        # 5) After the 2nd sync, both the DQN and target network are trained and having the same weights and biases again. This time, the target network can provide better outputs (based on the new experience) to guide the DQN learning.
        #
        # for state, action, new_state, reward, terminated in mini_batch: # get the data stored in the variable mini_batch
        # 
        #    
        #     # This if-else block implements the DQN Target Formula:
        #     if terminated: # if the game is over
        #         target = reward # the target (ground truth) reward = reward
        # 
        #    else:  # if the game is not over
        #         with torch.no_grad:
        #             target_q = reward + self.discount_factor_g * target_dqn(new_state).max() # the target (ground truth) reward
        # 
        #

        # Transpose the list of experience and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch) # Split the elements (state, action, new_state, reward, terminated) of the sampled experience instances stored in the variable mini_batch into different variables (states, actions, new_states, rewards, terminations) respectively. Since we set mini_batch_size=32, there are 32 sampled experience instances stored in the variable mini_batch. Hence, each of the (states, actions, new_states, rewards, terminations) variable will have 32 items/entries (length). Since the variable mini_batch is a tensor object, the resulting variables (states, actions, new_states, rewards, terminations) are also tensor objects, so that each entry of the resulting variables is a tensor such that {row/index 1: tensor1; row/index 2: tensor2..., where tensor1 = tensor([value1,value2,value3,value4,...]) }.
        # tensor object refers to the type of a variable. If a variable is a tensor object, then that variable stores a sequence of tensors, while each row of that variable is a tensor.
        # We use torch.stack() to join (concatenate) a sequence of tensors (two or more tensors) along a new dimension. The tensors must be same dimensions and shape. It inserts a new dimension and concatenates the tensors along that dimension. Reference: https://www.geeksforgeeks.org/python-pytorch-stack-method/
        
        # Stack tensors (in the form of sequence, one-by-one) of the tensor object called states to create batch tensors (EG: from {row/index 1: tensor2; row/index 2: tensor2..., where tensor1 = tensor([value1,value2,value3,value4,...]) } into tensor([[tensor1, tensor2, tensor3, ...]])), then store in tensor object states.
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)

        # Use torch.tensor() to convert each boolean string (True or False) in the variable terminations into a float number (1.0 or 0.0). We send "torch.tensor(terminations).float()" to the selected device to do process (to do the conversion and the results will be returned as a tensor object). We do this conversion so that at later stage we can use the terminations information to do math directly.
        terminations = torch.tensor(terminations).float().to(device)

        # After stacking all the tensors of each element of the memory into a batch tensor, we can use PyTorch to calculate the Q values for each tensor in the batch tensor in one-shot (instead of calculating the Q value of a tensor (a sampled experience instance) one at a time, then repeat the calculation for 32 times to get the Q values of all the sampled experience instances).

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states) ==> tensor([[1,2,3],[4,5,6]])
                .max(dim=1)            ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3,0,0,1]))
                [0]                    ==> tensor([3,6])
            '''


        # Calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze() # get the predicted Q values based on the state information
        '''
            policy_dqn(states)                              ==> tensor([[1,2,3],[4,5,6]])
            .gather(dim=1, index=actions.unsqueeze(dim=1))
            .squeeze()
        '''


        # Compute loss for the whole mini batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad() # Clear gradients
        loss.backward() # Compute gradients (backpropagation)
        self.optimizer.step() # Update DQN parameters (EG: weights and bias)

# Define the main function (analogous to the body of a script). The functions defined outside this main function play the supporting role when they are called inside this main function.            
if __name__ == '__main__' :
    
    # Parse command line inputs
    parser = argparse.ArgumentParser(description = 'Train or test model.')
    parser.add_argument('hyperparameters', help='') # if want to run in the DQN training mode, pass in the hyperparamter set name available in hyperparameters.yml.
    parser.add_argument('--train', help='Training mode', action = 'store_true') # if want to run in the DQN training mode, pass in '--train'. Else, we will run the DQN testing mode.
    args = parser.parse_args() # Need to self learn how to use ArgumentParser() & parse_args() so that we can directly run the file by providing the arguments we set at Command Prompt terminal

    dql = Agent(hyperparameter_set=args.hyperparameters) # pass the specified hyperparamter set name to the __init__() of Agent class function

    if args.train:
        dql.run(is_training = True)  # Execute the run() of Agent class function, by performing DQN training, DON't show the environment on a figure window.

    else:
        dql.run(is_training = False, render = True)  # Execute the run() of Agent class function, by performing DQN testing, show the environment on a figure window.


#**Important Instructions to run the DQN training or testing with a specified environment:**
# 1) For cartpole environment:
#     1) To perform DQN training:
#        1) Method 1 (Using Command Prompt terminal, not run in debug mode) [Note: you can create multiple Command Prompt terminal to perform DQN training or testing with different hyperparameters simultaneously]:
#            1) Open a new Command Prompt terminal
#            2) Enter the command "python D:/AI_Master_New/Under_Local_Git_Covered/Deep_Learning_Tutorials_codebasics/DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode/agent.py cartpole1 --train" # the syntax is "python MainFile_AbsolutePath HyperparamterSetName_AvailableInHyperparamtersYAMLFile --train"
#        2) Method 2 (using JSON file, run in debug mode):
#            1) Open a new JSON file at "Run and Debug" section (skip this step you have created the JSON file to run the DQN training)
#            2) Customize the JSON file according to your needs (EG: the name of this configuration JSON file, the absolute path of the file you want to run, pass in the arguments as requested by "parser.add_argument" in the program (the file you want to run))
#            3) Select the name of the JSON file you want and run it at "Run and Debug" section 
#    
#    2) To perform DQN testing:
#        1) Method 1 (Using Command Prompt terminal, not run in debug mode) [Note: you can create multiple Command Prompt terminal to perform DQN training or testing with different hyperparameters simultaneously]:
#            1) Open a new Command Prompt terminal
#            2) Enter the command "python D:/AI_Master_New/Under_Local_Git_Covered/Deep_Learning_Tutorials_codebasics/DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode/agent.py cartpole1" # the syntax is "python MainFile_AbsolutePath HyperparamterSetName_AvailableInHyperparamtersYAMLFile"
#        2) Method 2 (using JSON file, run in debug mode):
#            1) Open a new JSON file at "Run and Debug" section (skip this step you have created the JSON file to run the DQN training)
#            2) Customize the JSON file according to your needs (EG: the name of this configuration JSON file, the absolute path of the file you want to run, pass in the arguments as requested by "parser.add_argument" in the program (the file you want to run))
#            3) Select the name of the JSON file you want and run it at "Run and Debug" section 
#
#  2) For Flappy Bird environment:
#     1) To perform DQN training:
#        1) Method 1 (Using Command Prompt terminal, not run in debug mode) [Note: you can create multiple Command Prompt terminal to perform DQN training or testing with different hyperparameters simultaneously]:
#            1) Open a new Command Prompt terminal
#            2) Enter the command "python D:/AI_Master_New/Under_Local_Git_Covered/Deep_Learning_Tutorials_codebasics/DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode/agent.py flappybird1 --train" # the syntax is "python MainFile_AbsolutePath HyperparamterSetName_AvailableInHyperparamtersYAMLFile --train"
#        2) Method 2 (using JSON file, run in debug mode):
#            1) Open a new JSON file at "Run and Debug" section (skip this step you have created the JSON file to run the DQN training)
#            2) Customize the JSON file according to your needs (EG: the name of this configuration JSON file, the absolute path of the file you want to run, pass in the arguments as requested by "parser.add_argument" in the program (the file you want to run))
#            3) Select the name of the JSON file you want and run it at "Run and Debug" section 
#    
#    2) To perform DQN testing:
#        1) Method 1 (Using Command Prompt terminal, not run in debug mode) [Note: you can create multiple Command Prompt terminal to perform DQN training or testing with different hyperparameters simultaneously]:
#            1) Open a new Command Prompt terminal
#            2) Enter the command "python D:/AI_Master_New/Under_Local_Git_Covered/Deep_Learning_Tutorials_codebasics/DeepQLearning_DQN_PyTorch_Beginners_Tutorial_JohnnyCode/agent.py flappybird1" # the syntax is "python MainFile_AbsolutePath HyperparamterSetName_AvailableInHyperparamtersYAMLFile"
#        2) Method 2 (using JSON file, run in debug mode):
#            1) Open a new JSON file at "Run and Debug" section (skip this step you have created the JSON file to run the DQN training)
#            2) Customize the JSON file according to your needs (EG: the name of this configuration JSON file, the absolute path of the file you want to run, pass in the arguments as requested by "parser.add_argument" in the program (the file you want to run))
#            3) Select the name of the JSON file you want and run it at "Run and Debug" section 
#                
# **The information obtained at the first iteration of the loop, through debug console (refers to D:\AI_Master_New\Under_Local_Git_Covered\Deep_Learning_Tutorials_codebasics\DeepQLearning(DQN)_PyTorch_Beginners_Tutorial_JohnnyCode\Implement_Deep_Q_Learning_with_PyTorch_and_Train_Flappy_Bird_DQN_PyTorch_Beginners_Tutorial1\Explain (don't run)\hidden\photo1.png):**
# 1) The action selected is 0, means the flappy bird do nothing.
# 2) After the action is executed, the observation of 12 different state information is stored in obs vector as image above. The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1.
# 3) The env.observation_space shows that each element value is normalized to the range between -1 and 1, while the size of the observation space is 12 (that's why the obs vector consists of 12 elements). Observation space refers to all the state information gathered in an iteration (when the agent arrives at next state after executing the action in existing iteration/state)
# 4) The reward the agent gained when it arrives at the next state after executing the action in existing iteration/state is 0.1. According to the Flappy Bird environment GitHub repository, the reward of 0.1 means the agent (flappy bird) stays alive when it arrives at the next state.

# Watched until 5:32