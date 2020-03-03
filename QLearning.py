from setup import powerGrid_ieee2
import pandas as pd
import numpy as np
import math
import pickle
import os
import matplotlib.pyplot as plt
import copy
from datetime import datetime

class qLearning:
    def __init__(self, learningRate, decayRate, numOfEpisodes, stepsPerEpisode, epsilon ,annealingConstant, annealAfter):
        self.voltageRanges=['<0.75','0.75-0-79','0.8-0-84','0.85-0.89','0.9-0.94','0.95-0.99','1-1.04','1.05-1.09','1.1-1.14','1.15-1.19','1.2-1.24','>=1.25'];
        self.voltageRanges_2=['<0.85','0.85-0.874','0.875-0.899','0.9-0.924','0.925-0-949','0.95-0.974','0.975.0.999','1-1.024','1.025-1.049','1.05-1.074','1.075-1.1','>=1.1'];
        self.loadingPercentRange=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109','110-119','120-129','130-139','140-149','150 and above'];
        self.statesLev1 = ['v_' + x + '_l_' + y for x in self.voltageRanges_2 for y in self.loadingPercentRange]
        self.states=['s1:' + x + ';s2:' + y +';' for x in self.statesLev1 for y in self.statesLev1]

        self.env_2bus=powerGrid_ieee2();
        self.actions=['v_ref:'+str(x)+';lp_ref:'+str(y) for x in self.env_2bus.actionSpace['v_ref_pu'] for y in self.env_2bus.actionSpace['lp_ref']]
        self.checkPointName='pickled_q_table_lr'+str(learningRate)+'dr'+str(decayRate)+'noe'+str(numOfEpisodes)+'spe'+str(stepsPerEpisode)+'e'+str(epsilon)+'ac'+str(annealingConstant)+'aa'+str(annealAfter)+'.pkl';
        if os.path.isfile(self.checkPointName):
            print('loading data from checkpoint')
            with open(self.checkPointName, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                self.epsilon = data['e'];
                self.q_table = data['q_table'];
                self.allRewards = data['allRewards']
        else:
            self.q_table = pd.DataFrame(0, index=np.arange(len(self.actions)), columns=self.states);
            self.epsilon = epsilon
            self.allRewards = [];
        self.numOfEpisodes = numOfEpisodes
        self.annealingRate = annealingConstant
        self.numOfSteps = stepsPerEpisode
        self.learningRate = learningRate
        self.decayRate = decayRate
        self.annealAfter=annealAfter

    def getStateFromMeasurements(self, voltageLoadingPercentArray):
        res='';
        count=1;
        for voltageLoadingPercent in voltageLoadingPercentArray:
            voltage=voltageLoadingPercent[0];
            loadingPercent=voltageLoadingPercent[1];
            v_ind = (math.floor(round(voltage * 100) / 5) - 14);
            v_ind = v_ind if v_ind >=0 else 0;
            v_ind = v_ind if v_ind < len(self.voltageRanges) else len(self.voltageRanges)-1;
            l_ind = (math.floor(round(loadingPercent)/10));
            l_ind = l_ind if l_ind < len(self.loadingPercentRange) else len(self.loadingPercentRange)-1;
            res=res + 's'+str(count)+':v_' + self.voltageRanges[v_ind] + '_l_' + self.loadingPercentRange[l_ind]+';';
            count+=1
        return res;

    def getStateFromMeasurements_2(self, voltageLoadingPercentArray):
        res = '';
        count = 1;
        for voltageLoadingPercent in voltageLoadingPercentArray:
            voltage = voltageLoadingPercent[0];
            loadingPercent = voltageLoadingPercent[1];
            v_ind = (math.floor(round(voltage * 100) / 2.5) - 33);
            v_ind = v_ind if v_ind >= 0 else 0;
            v_ind = v_ind if v_ind < len(self.voltageRanges_2) else len(self.voltageRanges_2) - 1;
            l_ind = (math.floor(round(loadingPercent) / 10));
            l_ind = l_ind if l_ind < len(self.loadingPercentRange) else len(self.loadingPercentRange) - 1;
            res = res + 's' + str(count) + ':v_' + self.voltageRanges_2[v_ind] + '_l_' + self.loadingPercentRange[
                l_ind] + ';';
            count += 1
        return res;

    def getActionFromIndex(self, ind):
        actionString=self.actions[ind];
        actionStringSplitted=actionString.split(';');
        voltage = actionStringSplitted[0].split(':')[1];
        loadingPercent = actionStringSplitted[1].split(':')[1];
        return((int(loadingPercent),float(voltage) ));

    def test(self):
        rewards=[]
        self.env_2bus.reset();
        count=0;
        for i in self.q_table:
           if len(self.q_table[i].unique()) > 1:
               print(i);
               count+=1;
        print('Number of Unique States Visited: '+ str(count));
          # print(q_table[i].min())
        currentMeasurements = self.env_2bus.getCurrentState();
        oldMeasurements=currentMeasurements;
        for i in range(0,12):
            currentState = self.getStateFromMeasurements_2([oldMeasurements,currentMeasurements]);
            actionIndex = self.q_table[currentState].idxmax();
            #print(q_table[currentState].unique())
            action = self.getActionFromIndex(actionIndex);
            oldMeasurements=currentMeasurements;
            currentMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1]);
            print(self.env_2bus.net.res_bus.vm_pu)
            print(self.env_2bus.net.res_line)
            rewards.append(reward);
        print(sum(rewards))
        plt.plot(list(range(0, len(rewards))), rewards)
        plt.show();

    def testAllActions(self):
        rewards = []
        compensation = []
        measurements = []
        self.env_2bus.reset();
        #print(env_2bus.net.load)
        print(self.env_2bus.net.res_bus.vm_pu[1])
        print(self.env_2bus.net.res_line.loading_percent[1])
        print(self.env_2bus.net.res_line.loading_percent[0])
        print(np.std(self.env_2bus.net.res_line.loading_percent))
        for i in range(0, len(self.actions)):
            copyNetwork=copy.deepcopy(self.env_2bus);
            #measurements.append({'v_meas': copyNetwork.net.res_bus.vm_pu[1], 'lp_meas': copyNetwork.net.res_line.loading_percent[1]})
            #copyNetwork.runEnv()
            #print(copyNetwork.net.load)
            #copyNw=powerGrid_ieee2();
            #copyNw.stateIndex=env_2bus.stateIndex; #Copying stateindex, but load is still radomized from init() to different value than env_2bus
            #copyNw.scaleLoadAndPowerValue(copyNw.stateIndex, -1);
            #copyNw.runEnv()
            #print(copyNw.net.load)
            action = self.getActionFromIndex(i);
            nextStateMeasurements, reward, done = copyNetwork.takeAction(action[0], action[1]);
            rewards.append(reward);
            compensation.append({'k':copyNetwork.k_old,'q': copyNetwork.q_old})
            measurements.append({'v_bus1': copyNetwork.net.res_bus.vm_pu[1], 'lp_std': np.std(copyNetwork.net.res_line.loading_percent)})
        currentMeasurements = self.env_2bus.getCurrentState();
        currentState = self.getStateFromMeasurements_2([currentMeasurements,currentMeasurements]);
        actionIndex = self.q_table[currentState].idxmax();
        #action = getActionFromIndex(actionIndex);
        #nextStateMeasurements2, reward2, done2 = env_2bus.takeAction(action[0], action[1]);
        #print(env_2bus.stateIndex-1)
        #print(reward2);
        #print(rewards)
        print('Reward From Greedy Action '+self.actions[actionIndex]+' : '+str(rewards[actionIndex]))
        print(compensation[actionIndex])
        print('Max Reward Possible: '+str(max(rewards))+' at action: '+self.actions[rewards.index(max(rewards))]);
        print(compensation[rewards.index(max(rewards))])
        print(measurements[actionIndex])

    def train(self):
            print('epsilon: ' + str(self.epsilon))
            print('Has already been  trained for following num of episodes: ' + str(len(self.allRewards)))
            for i in range(0,self.numOfEpisodes):
                accumulatedReward=0;
                self.env_2bus.reset();
                currentMeasurements = self.env_2bus.getCurrentState();
                oldMeasurements = currentMeasurements;
                for j in range(0,self.numOfSteps):
                    epsComp = np.random.random();
                    currentState=self.getStateFromMeasurements_2([oldMeasurements,currentMeasurements]);
                    if epsComp <= self.epsilon:
                        # Exploration Part
                        actionIndex = np.random.choice(99, 1)[0]
                    else:
                        # Greedy Approach
                        actionIndex=self.q_table[currentState].idxmax();
                    action = self.getActionFromIndex(actionIndex);
                    oldMeasurements=currentMeasurements;
                    currentMeasurements, reward, done = self.env_2bus.takeAction(action[0],action[1])
                    accumulatedReward += reward;
                    if done:
                        nextStateMaxQValue = 0;
                    else:
                        nextState = self.getStateFromMeasurements_2([oldMeasurements,currentMeasurements]);
                        nextStateMaxQValue=self.q_table[nextState].max();
                    self.q_table.iloc[actionIndex,self.states.index(currentState)] = self.q_table[currentState][actionIndex] + self.learningRate*(reward + self.decayRate*nextStateMaxQValue - self.q_table[currentState][actionIndex])
                    if done:
                        break;
                self.allRewards.append(accumulatedReward);
                if (i+1) %100 == 0:
                    print('Episode: '+str(i+1)+'; reward:'+str(accumulatedReward))
                if (i+1) % self.annealAfter == 0:
                    self.epsilon=self.annealingRate*self.epsilon;
                    print('saving checkpoint data')
                    pickledData={'q_table':self.q_table, 'e':self.epsilon, 'allRewards':self.allRewards}
                    pickle.dump(pickledData, open(self.checkPointName, "wb"))


