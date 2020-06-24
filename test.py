from QLearning import qLearning
from DQN import DQN
from TD3 import TD3
# import threading
# import time
# import logging
from setup import powerGrid_ieee2


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

#l9 = qLearning(learningRate = 0.001, decayRate=0.7, numOfEpisodes=25000,stepsPerEpisode=24, epsilon=1, annealingConstant=0.98,annealAfter=200,checkpoint='pickled_q_table_lr0.001dr0.7noe50000spe75e1ac0.98aa400.pkl');
#print(ql1.states[0])

#ql10=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=180000,stepsPerEpisode=2,epsilon=1,annealingConstant=0.98,annealAfter=1500,
#               checkpoint='pickles_qlearning\pickled_q_table_lr0.001dr0.7noe50000spe75e1ac0.98aa400.pkl');
#ql10.train()
#ql5.comparePerformance(steps=50, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1)
#ql5.test(100,1)

#print(ql1.getStateFromMeasurements_2([(0.85,9),(0.95,9)]))
#ql2=qLearning(0.001,0.8,2000,12,1,0.98,150);
#ql3=qLearning(0.001,0.7,2000,12,1,0.98,150);
#ql2.train();
#ql2.test()
#ql2.testAllActions()
# try:
#     threads = []
#     x = threading.Thread(target=ql1.train)
#     threads.append(x)
#     x.start();
#     # y = threading.Thread(target=ql2.train)
#     # threads.append(y)
#     # y.start();
#     # y = threading.Thread(target=ql3.train)
#     # threads.append(y)
#     # y.start();
#     for index, thread in enumerate(threads):
#         logging.info("Main    : before joining thread %d.", index)
#         thread.join()
#         logging.info("Main    : thread %d done", index)
# except:
#    print("Error: unable to start thread")

################### DQN ###########################
############## done training ################
#dqn4=DQN(2, 0.001, 3000, 128, 0.7, 50000, 24, 1, 0.99, 200,1000)
#dqn5=DQN(2, 0.001, 3000, 32, 0.7, 50000, 6, 1, 0.99, 200,1000)
#dqn6=DQN(2, 0.001, 2000, 32, 0.6, 100000, 6, 1, 0.99, 200,1000)
############## end block ################
#dqn7=DQN(2, 0.001, 3000, 32, 0.6, 50000, 6, 1, 0.99, 200,1000)
#dqn7.train()
#dqn4.test(10,24,1)


# Create load profile graph and joined graph  for report
#noRL = powerGrid_ieee2('dqn')
#noRL.setMode('test')
#noRL.runNoRL(1)


#New prints for RL result:
#qlres=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
#dqnres=DQN(2, 0.001, 2000, 64, 0.7, 10000, 12, 1, 0.98, 100, 200, True)
td3res=TD3(2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)
td3res.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)

# WINNERS:
#qlres=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
#dqnres=DQN(2, 0.001, 2000, 64, 0.7, 10000, 12, 1, 0.98, 100, 200, True)
#td3res=TD3(2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)