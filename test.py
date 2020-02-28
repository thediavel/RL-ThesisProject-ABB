from setup import powerGrid_ieee4,powerGrid_ieee2
import pickle
import  matplotlib.pyplot as plt
#env_4bus=powerGrid_ieee4();
with open('pickled_q_table.pkl', 'rb') as pickle_file:
    data = pickle.load(pickle_file)
    epsilon = data['e'];
    q_table = data['q_table'];
    allRewards=data['allRewards'];

#plt.plot(list(range(0,len(allRewards))),allRewards)
print(epsilon)




#plt.scatter(list(range(0,len(allRewards))),allRewards)
#plt.show()

#for i in q_table:
#    print(q_table[i].min())

env_2bus=powerGrid_ieee2();
for i in range(0,0):
   result, reward, done = env_2bus.takeAction(80, 0.75)
   ## more reward means better solution
   print(reward)
   print(result)
   if done:
       env_2bus.reset();


"""print(len(set(env.powerProfile)))
print(len(set(env.loadProfile)))
env.plotGridFlow()
x = venv.loadProfile;
num_bins = 100
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()
print(sum(x)/len(x))
"""