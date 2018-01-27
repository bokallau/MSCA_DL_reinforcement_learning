#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:05:46 2018

@author: bruceokallau

Q learning example from Arthur Juliani, with extra comments and changes where noted.

"""

import gym
import numpy as np
import matplotlib.pyplot as plt

# load environment
env = gym.make('FrozenLake-v0')
'''
The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

FrozenLake-v0 defines "solving" as getting average reward of 0.78 over 100 consecutive trials.

the surface is described using a grid like the following:

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

BO note: This can be thought of similar to a maze except instead of running into
a wall the agent falls into the water and the trail run is over.
It must be noted that the agent will not always move to the square it chooses because of the slippery nature of ice. This incorporates randomness. But from my understanding the location of the starting point, goal and holes remain the same
for each trial. This encourages the agent to use its memory to find a solution.
'''

# impliment q-learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    jList.append(j)
    rList.append(rAll)
    
print ("Score over time: " +  str(sum(rList)/num_episodes))
# Score over time: 0.5755
print("Average number of steps per trial: " +  str(sum(jList)/num_episodes))
#Average number of steps per trial: 39.884

print ("Final Q-Table Values")
print (Q)

lines = plt.plot(rList)
l1 = lines
plt.setp(l1,linewidth=.1, color='g')
steps = np.array(jList)/100
l2 = plt.plot(steps)
plt.setp(l2,linewidth=.1, color='b')
plt.show()
'''
BO note:
reduced the width of the lines significantly to allow the failures to be displayed as well.
you can see the agent starting to reach the goal around 125 trials but there's still many failurs among the sucesses, which is why the score over time is only 57.55%
Trying to superimpose the number of steps to see if we might get more sucesses if we allowed the agent to take more than 99 steps but the plot is hard to read.
'''
