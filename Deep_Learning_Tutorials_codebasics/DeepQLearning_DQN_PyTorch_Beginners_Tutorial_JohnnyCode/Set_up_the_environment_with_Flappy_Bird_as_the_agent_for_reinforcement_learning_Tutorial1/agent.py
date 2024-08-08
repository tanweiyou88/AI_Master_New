"""
# Set up the environment with Flappy Bird as the agent for reinforcement learning (DQN PyTorch Beginners Tutorial 1, by Johny Code)

Link to the Youtube video tutorial: https://www.youtube.com/watch?v=arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=2 <br />
Link to the Gymnasium official website: https://gymnasium.farama.org/ <br />
Link to the Flappy Bird environment GitHub repository: https://github.com/markub3327/flappy-bird-gymnasium <br />

**Instructions to conduct this tutorial:**
1) Must use the virtual environment called "dqnenv", because this is the only environment installed with Flappy Bird environment on this device.
2) Must run "agent.py". "agent.ipynb" is only used to provide explanation on "agent.py" in an arranged way, but it cannot shows the game on screen.
3) On agent.py, can press F5 to conduct "Run and debug", add breakpoint to certain line, and use the debug console (by providing the name of the variables you interested), to observe the information stored in different variables at different instances (at different lines and different iteration).
"""

import flappy_bird_gymnasium # Import the flappy bird environment, which is compatible with gymnasium
import gymnasium # Import the gymnasium, which is an API standard for reinforcement learning with a diverse collection of reference environments

# Create an instance of the flappy bird environment called env, using the "FlappyBird-v0" environment model and with the help of gymnasium. The parameter "render_mode="human"" is used to render the game on the screen. The parameter "use_lidar=False" is the custom parameter that allow the user to turn on (use the information of) the LIDAR sensor or off (don't use the information of LIDAR).
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) # This means env is the environment that we create here.
## Just for extra illustration of different environment, can ignore/comment the line below.
# env = gymnasium.make("CartPole-v1", render_mode="human") # This means env is the environment that we create here. 

obs, _ = env.reset() # call the reset() to initialize the environment

while True: # This is an infinite loop for the agent(flappy bird) to interact with the environment through action, observation, and reward.
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() # use sample() on action space(action_space) of the environment(env) to get a random action for the agent (flappy bird) at each iteration. This means the sample() might return different actions(values) at each iteration. 
    # More explanation: 
    # 1) The action_space of an environment refers to all possible actions that can be taken by the agent. In this environment, the action_space consists of only 2 values(0:the agent[flappy bird] do nothing; 1:the agent[flappy bird] flaps its wing to fly up). 
    # 2) Hence, the sample() here will only return either 0 or 1. While the returned value is stored in the action variable.


    # Processing:
    obs, reward, terminated, _, info = env.step(action) # The action selected is passed into the step() to execute that action.
    # More explanation:
    # 1) The step() will give us back the observation (obs) [means what the next state is], reward [means how much reward we got from the last action],
    # terminated [terminated=True -> if the bird hits the ground or one of the pipes; else, terminated=False], _ means the parameter is not used,
    # info [just contains the additional information you can use for debugging or something]
    # 2) obs
    #   2.1) The obs is a vector that contains 12 elements, while each element represents a state information as a value (according to the repository, there are 12 different state information considered as observation). The 12 elements represent different state information respectively in sequence, according to the repository of the environemnt. Hence, the first element of the obs vector represents the state information of the last pipe's horizontal position,...., the last element of the obs vector represents the state information of the player's rotation.
    #   2.2) The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1 (because the author used this range to train the neural network)

    # Checking if the player is still alive
    if terminated: # if terminated=True
        break # we exit this infinite loop

env.close() # close the environment (the figure window)

# **The information obtained at the first iteration of the loop, through debug console (refers to D:\AI_Master_New\Under_Local_Git_Covered\Deep_Learning_Tutorials_codebasics\DeepQLearning(DQN)_PyTorch_Beginners_Tutorial_JohnnyCode\Implement_Deep_Q_Learning_with_PyTorch_and_Train_Flappy_Bird_DQN_PyTorch_Beginners_Tutorial1\Explain (don't run)\hidden\photo1.png):**
# 1) The action selected is 0, means the flappy bird do nothing.
# 2) After the action is executed, the observation of 12 different state information is stored in obs vector as image above. The elements' value of the obs vector are normalized (transformed into) to the range between -1 and 1.
# 3) The env.observation_space shows that each element value is normalized to the range between -1 and 1, while the size of the observation space is 12 (that's why the obs vector consists of 12 elements). Observation space refers to all the state information gathered in an iteration (when the agent arrives at next state after executing the action in existing iteration/state)
# 4) The reward the agent gained when it arrives at the next state after executing the action in existing iteration/state is 0.1. According to the Flappy Bird environment GitHub repository, the reward of 0.1 means the agent (flappy bird) stays alive when it arrives at the next state.