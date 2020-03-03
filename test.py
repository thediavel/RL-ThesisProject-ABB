from QLearning import qLearning
import threading
import time
import logging

ql1=qLearning(0.001,0.9,2000,12,1,0.98,150);
ql2=qLearning(0.001,0.8,2000,12,1,0.98,150);
ql1.train();
# try:
#     threads = []
#     x = threading.Thread(target=ql1.train)
#     threads.append(x)
#     x.start();
#     y = threading.Thread(target=ql2.train)
#     threads.append(y)
#     y.start();
#     for index, thread in enumerate(threads):
#         logging.info("Main    : before joining thread %d.", index)
#         thread.join()
#         logging.info("Main    : thread %d done", index)
# except:
#    print("Error: unable to start thread")