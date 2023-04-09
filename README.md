The repository has two functionalities. The first is very simple and was built more as
an exercise to ensure proof of concept. The second is more advanced and useful. This 
repository relies only on installation of numpy and matplotlib.pyplot.

1. Single simulation of a single player softmax MAB problem:
  This calculates and visualizes the accumulated reward against the round number 
  for a certain value of tau. In order to do this:
  - Access script: Single_Player_Softmax_MAB.py
  - Change MAB and Team values accordingly
  - Run script
  
2. Average of many simulations of a team with members defined by verbosity solving
  a MAB problem using different exploration-exploitation strategies: 
  This calculates and plots the accumulated reward, accumulated regret and rate of 
  choosing the best reward against the round number for different values of tau.
  In order to do this:
  - Access script: Averaged_Simulations.py
  - Change MAB, Team and Simulation values accordingly
  - Run script

