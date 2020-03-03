from setup import powerGrid_ieee2
import pandas as pd
import numpy as np
import math
import pickle
import os
import matplotlib.pyplot as plt
import copy

voltageRanges=['<0.75','0.75-0-79','0.8-0-84','0.85-0.89','0.9-0.94','0.95-0.99','1-1.04','1.05-1.09','1.1-1.14','1.15-1.19','1.2-1.24','>=1.25'];
voltageRanges_2=['<0.85','0.85-0.874','0.875-0.899','0.9-0.924','0.925-0-949','0.95-0.974','0.975.0.999','1-1.024','1.025-1.049','1.05-1.074','1.075-1.1','>=1.1'];
loadingPercentRange=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109','110-119','120-129','130-139','140-149','150 and above'];
states=['v:' + x + ';l:' + y for x in voltageRanges_2 for y in loadingPercentRange]
env_2bus=powerGrid_ieee2();
actions=['v_ref:'+str(x)+';lp_ref:'+str(y) for x in env_2bus.actionSpace['v_ref_pu'] for y in env_2bus.actionSpace['lp_ref']]


def getStateFromMeasurements(voltageLoadingPercent):
    voltage=voltageLoadingPercent[0];
    loadingPercent=voltageLoadingPercent[1];
    v_ind = (math.floor(round(voltage * 100) / 5) - 14);
    v_ind = v_ind if v_ind >=0 else 0;
    v_ind = v_ind if v_ind < len(voltageRanges) else len(voltageRanges)-1;
    l_ind = (math.floor(round(loadingPercent)/10));
    l_ind = l_ind if l_ind < len(loadingPercentRange) else len(loadingPercentRange)-1;
    return 'v:'+voltageRanges[v_ind]+';l:'+loadingPercentRange[l_ind];

def getStateFromMeasurements_2(voltageLoadingPercent):
    voltage=voltageLoadingPercent[0];
    loadingPercent=voltageLoadingPercent[1];
    v_ind = (math.floor(round(voltage * 100) / 2.5) - 33);
    v_ind = v_ind if v_ind >=0 else 0;
    v_ind = v_ind if v_ind < len(voltageRanges_2) else len(voltageRanges_2)-1;
    l_ind = (math.floor(round(loadingPercent)/10));
    l_ind = l_ind if l_ind < len(loadingPercentRange) else len(loadingPercentRange)-1;
    return 'v:'+voltageRanges_2[v_ind]+';l:'+loadingPercentRange[l_ind];

def getActionFromIndex(ind):
    actionString=actions[ind];
    actionStringSplitted=actionString.split(';');
    voltage = actionStringSplitted[0].split(':')[1];
    loadingPercent = actionStringSplitted[1].split(':')[1];
    return((int(loadingPercent),float(voltage) ));


if os.path.isfile('pickled_q_table.pkl'):
    print('loading data from checkpoint')
    with open('pickled_q_table.pkl', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        epsilon = data['e'];
        q_table=data['q_table'];
        allRewards=data['allRewards']
else:
    q_table = pd.DataFrame(0, index=np.arange(len(actions)), columns=states);
    epsilon = 1
    allRewards = [];
numOfEpisodes = 30000
annealingRate=0.98
numOfSteps = 12
learningRate = 0.001
decayRate = 0.9
#print(q_table['v:0.75-0-79;l:120-129'][0])
print('epsilon: '+str(epsilon))
print('Has already been  trained for following num of episodes: '+str(len(allRewards)))

def test(q_table):
    rewards=[]
    env_2bus.reset();
    count=0;
    for i in q_table:
       if len(q_table[i].unique()) > 1:
           print(i);
           count+=1;
    print('Number of Unique States Visited: '+ str(count));
      # print(q_table[i].min())
    for i in range(0,0):
        currentMeasurements = env_2bus.getCurrentState();
        currentState = getStateFromMeasurements_2(currentMeasurements);
        actionIndex = q_table[currentState].idxmax();
        #print(q_table[currentState].unique())
        action = getActionFromIndex(actionIndex);
        nextStateMeasurements, reward, done = env_2bus.takeAction(action[0], action[1]);
        print(env_2bus.net.res_bus.vm_pu)
        print(env_2bus.net.res_line)
        rewards.append(reward);
    print(sum(rewards))
    plt.plot(list(range(0, len(rewards))), rewards)
    plt.show();

def testAllActions(q_table):
    rewards = []
    compensation = []
    measurements = []
    env_2bus.reset();
    #print(env_2bus.net.load)
    print(env_2bus.net.res_bus.vm_pu[1])
    print(env_2bus.net.res_line.loading_percent[1])
    print(env_2bus.net.res_line.loading_percent[0])
    print(np.std(env_2bus.net.res_line.loading_percent))
    for i in range(0, len(actions)):
        copyNetwork=copy.deepcopy(env_2bus);
        #measurements.append({'v_meas': copyNetwork.net.res_bus.vm_pu[1], 'lp_meas': copyNetwork.net.res_line.loading_percent[1]})
        #copyNetwork.runEnv()
        #print(copyNetwork.net.load)
        #copyNw=powerGrid_ieee2();
        #copyNw.stateIndex=env_2bus.stateIndex; #Copying stateindex, but load is still radomized from init() to different value than env_2bus
        #copyNw.scaleLoadAndPowerValue(copyNw.stateIndex, -1);
        #copyNw.runEnv()
        #print(copyNw.net.load)
        action = getActionFromIndex(i);
        nextStateMeasurements, reward, done = copyNetwork.takeAction(action[0], action[1]);
        rewards.append(reward);
        compensation.append({'k':copyNetwork.k_old,'q': copyNetwork.q_old})
        measurements.append({'v_bus1': copyNetwork.net.res_bus.vm_pu[1], 'lp_std': np.std(copyNetwork.net.res_line.loading_percent)})
    currentMeasurements = env_2bus.getCurrentState();
    currentState = getStateFromMeasurements_2(currentMeasurements);
    actionIndex = q_table[currentState].idxmax();
    #action = getActionFromIndex(actionIndex);
    #nextStateMeasurements2, reward2, done2 = env_2bus.takeAction(action[0], action[1]);
    #print(env_2bus.stateIndex-1)
    #print(reward2);
    #print(rewards)
    print('Reward From Greedy Action '+actions[actionIndex]+' : '+str(rewards[actionIndex]))
    print(compensation[actionIndex])
    print('Max Reward Possible: '+str(max(rewards))+' at action: '+actions[rewards.index(max(rewards))]);
    print(compensation[rewards.index(max(rewards))])
    print(measurements[actionIndex])

def train(numOfEpisodes, annealingRate,epsilon, numOfSteps, learningRate, decayRate):
    for i in range(0,numOfEpisodes):
        accumulatedReward=0;
        env_2bus.reset();
        currentMeasurements = env_2bus.getCurrentState();
        for j in range(0,numOfSteps):
            epsComp = np.random.random();
            currentState=getStateFromMeasurements_2(currentMeasurements);
            if epsComp <= epsilon:
                # Exploration Part

                actionIndex = np.random.choice(99, 1)[0]
            else:
                # Greedy Approach
                actionIndex=q_table[currentState].idxmax();
            action = getActionFromIndex(actionIndex);
            nextStateMeasurements, reward, done = env_2bus.takeAction(action[0],action[1])
            accumulatedReward += reward;
            if done:
                nextStateMaxQValue = 0;
            else:
                nextState = getStateFromMeasurements_2(nextStateMeasurements);
                nextStateMaxQValue=q_table[nextState].max();
            #print(reward)
            #print(q_table[currentState][actionIndex] + learningRate * (reward + decayRate * nextStateMaxQValue - q_table[currentState][actionIndex]))
            q_table.iloc[actionIndex,states.index(currentState)] = q_table[currentState][actionIndex] + learningRate*(reward + decayRate*nextStateMaxQValue - q_table[currentState][actionIndex])
            #print(q_table[currentState][actionIndex])
            if done:
                break;
        allRewards.append(accumulatedReward);
        if (i+1) %10 == 0:
            print('Episode: '+str(i+1)+'; reward:'+str(accumulatedReward))
        if (i+1) % 150 == 0:
            epsilon=annealingRate*epsilon;
            print('saving checkpoint data')
            pickledData={'q_table':q_table, 'e':epsilon, 'allRewards':allRewards}
            pickle.dump(pickledData, open("pickled_q_table.pkl", "wb"))

train(numOfEpisodes, annealingRate,epsilon, numOfSteps, learningRate, decayRate);
#test(q_table);
#for i in range(0,10):
#    testAllActions(q_table)

#print(getStateFromMeasurements((0.77,123)))
#print(getActionFromIndex(56))
#print(q_table)
#createStateSpace()