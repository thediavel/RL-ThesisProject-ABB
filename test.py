from setup import powerGrid_ieee4,powerGrid_ieee2

#env_4bus=powerGrid_ieee4();
env_2bus=powerGrid_ieee2();
for i in range(0,2):
   result, reward, done = env_2bus.takeAction(0.6, 0.5)
   print(result)
   if done:
       env_2bus.reset();

#print(len(set(env.powerProfile)))
#print(len(set(env.loadProfile)))
#env.plotGridFlow()
# x = venv.loadProfile;
# num_bins = 100
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# plt.show()
# print(sum(x)/len(x))