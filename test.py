from QLearning import qLearning
import threading
import time
import logging

ql1=qLearning(0.001,0.9,2000,12,1,0.98,150);
#print(ql1.states[0])

#print(ql1.q_table['s1:v_1-1.024_l_0-9;s2:v_0.95-0.974_l_20-29;']);

#print(ql1.getStateFromMeasurements_2([(0.85,9),(0.95,9)]))
#ql2=qLearning(0.001,0.8,2000,12,1,0.98,150);
#ql3=qLearning(0.001,0.7,2000,12,1,0.98,150);
ql1.train();
#ql1.test()
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