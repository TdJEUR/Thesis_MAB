# Influence of Team Composition and Decision Making Strategies on Multi-Armed Bandit Problems

## Manual of the Repository 

Dependencies: this repository relies only on installation of numpy and matplotlib.pyplot. No other packages are utilised.

### 1. Influence of Composition of Team Verbosity and Strategy
Calculate and plot the accumulated reward, accumulated regret and rate of choosing the 
best reward against the round number for different values of tau for a team composed of 
members with assigned verbosity. In order to obtain a reliable result, the scenario 
is simulated multiple times and the average is taken.

Customizable Attributes are:
  - MAB problem settings:
    - Number of arms
    - Mean reward of each arm
    - Standard deviation of the reward of each arm
    - Number of rounds to play
  - Team settings:
    - Number of team members
    - Verbosity of each individual team member
    - Team exploitation-exploration strategies
  - Simulation settings:
    - Number of simulations to average

To utilise this function:
  - Access script: Averaged_Simulations.py
  - Change MAB, Team and Simulation values accordingly
  - Run script

Assumptions and Questions:
- The Softmax algorithm is used to map beliefs to probabilities.
- Regret is defined by not choosing the arm with the best mean, not the arm that 
would have output the best reward including standard deviation.
- Calculate individual choice probabilities from beliefs and combine these probabilities
instead of combining beliefs and then calculating a choice prob (does this make a difference?). 
- Method of combining team member choice probabilities.

### 2. Single simulation of a single player softmax MAB problem:
Calculate and visualize the accumulated reward against the round number 
for a certain value of tau. Used as proof of concept for main functionality.

Customizable Attributes are:
  - MAB problem settings:
    - Number of arms
    - Mean reward of each arm
    - Number of rounds to play
  - PLayer settings:
    - Verbosity of the player
    - Exploitation-exploration strategy
  
To utilise this function:
  - Access script: Single_Player_Softmax_MAB.py
  - Change MAB and PLayer values accordingly
  - Run script
  


