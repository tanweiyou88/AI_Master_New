import torch
from torch import nn
import torch.nn.functional as F

# **The standard way to create a neural network in pytorch is to define a class.**

# Define a class for the neural network, called DQN. The class DQN inherit the nn.
class DQN(nn.Module):

    # The __init__ function defines the layers of a neural network
    def __init__(self, state_dim, action_dim, hidden_dim = 256): # pass the arguments of input dimension = the number of features at the current state = the number of input neurons (nodes) in the input layer of the neural network (state_dim), output dimension = the number of actions in the action space of the environment = the number of output neurons (nodes) in the output layer of the neural network (action_dim), and the number of neurons in the hidden layer (here, we set the number of neurons in the hidden layer as 256)
        super(DQN, self).__init__()

        # In pytorch, the input layer of a neural network (the 1st layer of a neural network) is implicit (means we don't need to write code for the input layer)

        # Define the 1st hidden layer of the neural network (the second layer of the neural network). This layer is called fc1 in this script. [this layer is the Hidden Layer 1 shown in the video]
        self.fc1 = nn.Linear(state_dim, hidden_dim) # create a layer that applies a linear transformation to the incoming data, with input size of state_dim (or state_dim number of input features) and output size of hidden_dim (or hidden_dim number of output features)

        # Define the output layer of the neural network (the third layer of the neural network). This layer is called fc2 in this script.
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # create a layer that applies a linear transformation to the incoming data, with input size of hidden_dim (or hidden_dim number of input features) and output size of action_dim (or action_dim number of output features)

    # The forward function does the calculations within the neural network
    def forward(self, x): # x is the input that contains the 12 state values (the 12 features of the current state)
        x = F.relu(self.fc1(x)) # we provide the x (12 state values) to the fc1 layer (1st hidden layer of the neural network), followed by passing through the relu activation function. The output of the relu activation function of this fc1 layer is stored in x again.
        return self.fc2(x) # The output of the relu activation function of fc1 layer which is stored in x is provided to the fc2 layer (the output layer of the neural network) to calculate the Q values (or expected rewards that the agent receives if the agent takes the corresponding actions to move into next state, from the current state). The calculated Q values will be returned by this forward function.

# Define the main function (analogous to the body of a script). The functions defined outside this main function play the supporting role when they are called inside this main function.    
if __name__ == '__main__':
    state_dim = 12 # the state dimension (input dimension) has size of 12, because there are 12 state information considered in observation (according to the Flappy Bird GitHub repository). In other words, when the agent (flappy bird) took an action and reached a new state, the agent will get 12 state values at that new state, where each state value represents different information/meaning observed at that new state.
    action_dim = 2 # the action dimension (output dimension) has size of 2, because there are only 2 actions in the action space of the environemnt (according to the Flappy Bird GitHub repository)
    net = DQN(state_dim, action_dim) # declare the neural network (maybe use the __init__() in the DQN())
    state = torch.randn(1, state_dim) # create random input (total 12 random values will be provided, while each value represents a state value at the next(new) state after executing an action). If the flappy bird took 10 actions to move into 10 new states, we can simulate the 12 state values at each new state by having the code of "torch.randn(10, state_dim)". Then, we can pass the 12 state values at each new state = 12 * 10 = 120 state values in a batch to the neural network to calculate the corresponding Q values.
    output = net(state) # send the random input into the neural network to calculate the Q values. The calculated Q values provided at the output layer of the neural network is returned and stored in variable output. Only 2 Q values will be returned by the neural network because the output dimension of the neural network is defined as size of 2 (means there are only 2 output neurons in the output layer of the neural network, with each output neuron provide a Q value. A Q value represents the expected reward the agent will receive if the agent takes that action). (maybe use the forward() in the DQN()).
    print(output) # print the calculated Q values stored in variable output.