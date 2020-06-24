from QLearning import qLearning
from DQN import DQN
from TD3 import TD3
# import threading
# import time
# import logging
from setup import powerGrid_ieee2

#New prints for RL result:
#qlres=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
#dqnres=DQN(2, 0.001, 2000, 64, 0.7, 10000, 12, 1, 0.98, 100, 200, True)
td3res=TD3(2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)
td3res.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)

# WINNERS:
#qlres=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
#dqnres=DQN(2, 0.001, 2000, 64, 0.7, 10000, 12, 1, 0.98, 100, 200, True)
#td3res=TD3(2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)


########################################################################################################################
# Create load profile graph and joined graph for report
#noRL = powerGrid_ieee2('dqn')
#noRL.setMode('test')
#noRL.runNoRL(1)

## Some other RL algorithms that might be of interest:
################### Q LEARNING ###########################
############## done training ################
#ql1=qLearning(0.001,0.9,20100,12,1,0.98,150);
#ql2=qLearning(learningRate=0.001,decayRate=0.9,numOfEpisodes=24000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=200);
#ql3=qLearning(learningRate=0.001,decayRate=0.9,numOfEpisodes=30000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=250);
#ql4=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=20100,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=150);
#ql5=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=24000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=200);
#ql6=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=30000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=250);
#ql7=qLearning(learningRate=0.001,decayRate=0.9,numOfEpisodes=288000,stepsPerEpisode=1,epsilon=1,annealingConstant=0.98,annealAfter=2400);
#ql8=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=288000,stepsPerEpisode=1,epsilon=1,annealingConstant=0.98,annealAfter=2400);
############## end block ####################


################### DQN ###########################
############## done training ################
#dqn4=DQN(2, 0.001, 3000, 128, 0.7, 50000, 24, 1, 0.99, 200,1000)
#dqn5=DQN(2, 0.001, 3000, 32, 0.7, 50000, 6, 1, 0.99, 200,1000)
#dqn6=DQN(2, 0.001, 2000, 32, 0.6, 100000, 6, 1, 0.99, 200,1000)
############## end block ################
#dqn7=DQN(2, 0.001, 3000, 32, 0.6, 50000, 6, 1, 0.99, 200,1000)
#dqn7.train()
#dqn4.test(10,24,1)


