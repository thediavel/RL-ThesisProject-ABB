from QLearning import qLearning
# import threading
# import time
# import logging


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

#ql1=qLearning(learningRate=0.001,decayRate=0.5,numOfEpisodes=24000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=200);
#ql1=qLearning(learningRate=0.001,decayRate=0.5,numOfEpisodes=30000,stepsPerEpisode=12,epsilon=1,annealingConstant=0.98,annealAfter=250);

ql9 = qLearning(learningRate = 0.001, decayRate=0.7, numOfEpisodes=25000,stepsPerEpisode=24, epsilon=1, annealingConstant=0.98,annealAfter=200,checkpoint='pickled_q_table_lr0.001dr0.7noe50000spe75e1ac0.98aa400.pkl');

#print(ql1.states[0])
#ql2.comparePerformance(steps=24, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1)
#ql7=qLearning(learningRate=0.001,decayRate=0.9,numOfEpisodes=288000,stepsPerEpisode=1,epsilon=1,annealingConstant=0.98,annealAfter=2400);

ql9.train()
#ql9.test(2,10);
#ql2.test()
#ql1.compareWith([ql2],2,12)
#print(ql1.q_table['s1:v_1-1.024_l_0-9;s2:v_0.95-0.974_l_20-29;']);

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