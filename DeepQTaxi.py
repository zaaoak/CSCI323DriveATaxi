import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch as torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from collections import defaultdict
import gymnasium as gym
# Initialise the environment
import numpy as np
import pandas as pd
from typing import Callable, Optional, Self, Tuple
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import time
from itertools import product
# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)        # Second hidden layer (same size as h1)
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w
        
        self._init_weights()

    def _init_weights(self):
        pass
        # Xavier (Glorot) initialization for weights and zero bias
        # init.xavier_uniform_(self.fc1.weight)
        # init.zeros_(self.fc1.bias)
        # init.xavier_uniform_(self.out.weight)
        # init.zeros_(self.out.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output
        return x
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        actual_size = min(len(self.memory), sample_size)#if actual size smaller than sample size, then just use actual size instaed
        return random.sample(self.memory, actual_size)

    def __len__(self):
        return len(self.memory)
    def __str__(self):
        processed_memory = [
    (decode_int_state(mem[0]), action_names[mem[1]], decode_int_state(mem[2]), mem[3], mem[4]) 
    for mem in self.memory
]
        return '\n'.join(str(row) for row in processed_memory)

action_names = {
    0: "down",
    1: "up",
    2: "right",
    3: "left",
    4: "Pickup",
    5: "DropOff"
}

def decode_int_state(state: int):
    dest = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state % 5

    return taxi_col,taxi_row, passenger, dest#(x, y, passenger, destination)

env = gym.make('Taxi-v3', render_mode="ansi")

num_states = env.observation_space.n  # 500
num_actions = env.action_space.n       # 6 actions in Taxi


def encode_To_int_state( taxi_col, taxi_row,passenger_loc, destination=1):
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

def decode_int_state(state: int):
    dest = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state % 5

    return taxi_col,taxi_row, passenger, dest#(x, y, passenger, destination)

def encodeIntStateToDQNInput(state: int) -> torch.Tensor:
    taxi_col, taxi_row, passenger_loc, destination = decode_int_state(state)

    # One-hot encode each component
    taxi_col_oh = torch.zeros(5)
    taxi_col_oh[taxi_col] = 1.0

    taxi_row_oh = torch.zeros(5)
    taxi_row_oh[taxi_row] = 1.0

    passenger_oh = torch.zeros(5)
    passenger_oh[passenger_loc] = 1.0


    # Concatenate all parts into a single tensor
    state_vector = torch.cat([
        taxi_col_oh,
        taxi_row_oh,
        passenger_oh,
    ])

    return state_vector
def decode_DQN_input_to_tuple(input_vector):
    # input_vector shape: length 19 (5 + 5 + 5 + 4)
    
    # Decode taxi_col one-hot (first 5 elements)
    taxi_col_oh = input_vector[0:5]
    taxi_col = torch.argmax(taxi_col_oh).item()
    
    # Decode taxi_row one-hot (next 5 elements)
    taxi_row_oh = input_vector[5:10]
    taxi_row = torch.argmax(taxi_row_oh).item()
    
    # Decode passenger location one-hot (next 5 elements)
    passenger_oh = input_vector[10:15]
    passenger_loc = torch.argmax(passenger_oh).item()
    
    
    return (int(taxi_col), int(taxi_row), int(passenger_loc))

#because we only consider one drop off location which is hotel at 4,0, passenger being at either of the 4 location will be one hot encoded, the passenger pickup and drop off will be normalized
#first 2 nodes of input will be the X,Y normalized
#next 5 nodes will be the passenger location, one hot encoded, to be either at hotel 0,1,2,3 or in taxi(green)

#get all possible states and also possible inputs into our DQN, note we do not include location here, as we set that to be 1(green hotel)
x_range = range(5)         # 0 to 4
y_range = range(5)         # 0 to 4
passenger_values = [0,2,3, 4]  # we have to include all passenger locations, even though we only want pasenger at 1, because dropping off passenger at wrong hotel is allowed to change state, although get reward of -10
dest_values = [1]          # only 1
PossibleStates = list(product(x_range, y_range, passenger_values))
PossibleStatesEncoded = [encode_To_int_state(x,y,p,1) for x,y,p in PossibleStates]
PossibleStatesEncodedDQN = [encodeIntStateToDQNInput(x) for x in PossibleStatesEncoded]


class TaxiDQL():
    
    def __init__(self):
        self.learning_rate_a = 0.001
        self.discount_factor_g = 0.9
        self.network_sync_rate = 100
        self.replay_memory_size = 1000
        self.mini_batch_size = 32
        self.train_freq = 4
        self.loss_fn = nn.MSELoss()
        self.optimizer = None




    def train(self, episodes, render = False):
        StartState = encode_To_int_state(taxi_row=0, taxi_col=0, passenger_loc=0, destination=1)
        env = gym.make('Taxi-v3', render_mode="ansi")
        env.reset()
        env.unwrapped.s = StartState#set start state

        PolicyNet = DQN(15, 20, 6)
        PolicyNet.load_state_dict(torch.load("frozen_lake_dql.pt"))
        TargetNet = DQN(15, 20, 6)

        TargetNet.load_state_dict(PolicyNet.state_dict())#copy weights and biases from policy to target network
        #self.print_dqn(PolicyNet)
        epsilon = 1
        epsilon_decay = epsilon/episodes #decay epsilon down to zero towards the end of training
        rng = np.random.default_rng()
        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        
        self.optimizer = torch.optim.Adam(PolicyNet.parameters(), lr=self.learning_rate_a)
        memory = ReplayMemory(self.replay_memory_size)

        step_count = 0#this is for sync rate
        for i in range(episodes):
            env.reset()
            StartState = encode_To_int_state(taxi_row=random.randint(0, 4), taxi_col=random.randint(0, 4), passenger_loc=random.randint(0, 3), destination=1)
            env.unwrapped.s = StartState#set start state
            CurrentState = env.unwrapped.s#set current state
            terminated = False#reach terminal state?
            truncated = False#take more than 200 steps?

            #agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).

            while not terminated and not truncated:
                #agent will move around map

                #choose random action or best action (exploration or exploitation)
                if rng.random() <epsilon:
                    #do a random action
                    action = int(env.action_space.sample())#actions: 0=left , 1 = down, 2 = right, 3 = up
                else:
                    #take best action according to Policy network
                    with torch.no_grad():#no grad just prevents pytorch from calculating backpropagation values, as we just want the prediction
                        action = int(PolicyNet(encodeIntStateToDQNInput(CurrentState)).argmax().item())

                
                newState, reward, terminated, truncated, info = env.step(action)
                #print(f'action {action_names[action]}, reward: {reward}, state: {CurrentState} decoded: {decode_state(CurrentState)}, newstate: {newState} decoded: {decode_state(newState)}, DQN input: {encodeStateToDQNInput(CurrentState)}\n')

                #even if truncated, dont have to put anything here, because truncated is the program ending by itself, not the environment ending.
                memory.append((CurrentState, action, newState, reward, terminated))

                CurrentState = newState
                #increase step to know when to sync
                step_count += 1
                #add to reward
                rewards_per_episode[i] += reward
                #print(env.render())  # ← Show the current state as text after each step

            # Check if enough experience has been collected,sometimes might reach terminal state early in the very first episode so not alot of experience
            if len(memory)>self.mini_batch_size :
                mini_batch = memory.sample(self.mini_batch_size)
                #print(memory.memory)
                self.optimize(mini_batch, PolicyNet,TargetNet)  

                #self.print_dqn(PolicyNet)   

                # Decay epsilon
                epsilon = max(epsilon - epsilon_decay, 0)
                #epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    TargetNet.load_state_dict(PolicyNet.state_dict())
                    step_count=0
            #print(terminated or truncated)
            if(i%500 == 0 or i==episodes-1):
                print(f'episode: {i}, reward: {rewards_per_episode[i]}, epsilon: {epsilon}')
                #print(memory)
        self.print_dqn(PolicyNet)
        # Close environment
        env.close()

        # Save policy
        torch.save(PolicyNet.state_dict(), "frozen_lake_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(rewards_per_episode)
        
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):


        current_q_list = []
        target_q_list = []

        #sample an experience
        for state, action, new_state, reward, terminated in mini_batch:
            #print(decode_state(state), action, decode_state(new_state), reward, terminated)
            if terminated: 
                # Agent reached goal (reward=20) (if there are other terminal states, it will apply here)
            # When in a terminated state, target q value should be set to the reward.(reward + discount * futureQ ) future Q is zero for terminal state
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value using target network to approximate future Q value.
                with torch.no_grad():
                    #print(state, action, new_state, reward, terminated )
                    #print(f'tarhet {target_dqn(encodeIntStateToDQNInput(new_state)).max()}')
                    target_value = reward + self.discount_factor_g * target_dqn(encodeIntStateToDQNInput(new_state)).max()
                    #print(target_value)
                    target = torch.FloatTensor(target_value)

            # Get the current set of Q values
            current_q = policy_dqn(encodeIntStateToDQNInput(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            # Adjust the specific action to the target that was just calculated
            target_q = current_q.clone().detach()
            target_q[action] = target
            target_q_list.append(target_q)
            #print(current_q,target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        #print(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Print DQN: state, best action, q values
    def print_dqn(self, dqn,printall=False):
        sorted_states = sorted(
            PossibleStates,
            key=lambda t: (t[2], t[0], t[1])
        )
        statesToPrint =[
            (0,0,0),
            (0,0,4),
            (4,0,4),

        ]
        # Loop each state and print policy to console
        for x,y,passenger in sorted_states:
            #  Format q values for printing
            state = encode_To_int_state(x,y,passenger,1)
            q_values = ''
            for q in dqn(encodeIntStateToDQNInput(state)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = action_names[dqn(encodeIntStateToDQNInput(state)).argmax().item()]
            
            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            if((x,y,passenger) in statesToPrint) or printall:
                print(f'{int(x)},{int(y)},{passenger},{best_action},[{q_values}]', end='\n')         #print(f'{x},{y},{passenger},{best_action},[{q_values}]', end='\n')         
        print('\n--------------------------------------\n')
    def test(self, episodes ):
        # Create FrozenLake instance
        env = gym.make('Taxi-v3', render_mode="human")
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(7, 10,6) 
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn,True)

        for i in range(episodes):
            env.reset()
            env.unwrapped.s = StartState = encode_To_int_state(taxi_row=random.randint(0, 4), taxi_col=random.randint(0, 4), passenger_loc=random.randint(0, 3), destination=1)
            state =   StartState
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(encodeIntStateToDQNInput(state)).argmax().item()
                    #print(action)

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)
                

        env.close()




if __name__ == '__main__':
    new = TaxiDQL()
    new.train(10000)