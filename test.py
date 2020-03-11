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

#ql9 = qLearning(learningRate = 0.001, decayRate=0.7, numOfEpisodes=25000,stepsPerEpisode=24, epsilon=1, annealingConstant=0.98,annealAfter=200,checkpoint='pickled_q_table_lr0.001dr0.7noe50000spe75e1ac0.98aa400.pkl');
#print(ql1.states[0])

ql10=qLearning(learningRate=0.001,decayRate=0.7,numOfEpisodes=180000,stepsPerEpisode=2,epsilon=1,annealingConstant=0.98,annealAfter=1500);
ql10.train()

#ql9.test(2,3)
#ql10.comparePerformance(steps=40, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1)
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