from setup import powerGrid_ieee2
import pandas as pd
import numpy as np
import math
import pickle
import os
import matplotlib.pyplot as plt
import copy
import statistics as stat
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
        count=0;
        for i in self.q_table:
           if len(self.q_table[i].unique()) > 1:
               #print(i);
               count+=1;
        print('Number of Unique States Visited: '+ str(count));
          # print(q_table[i].min())
        for j in range(0,100):
            self.env_2bus.reset();
            currentMeasurements = self.env_2bus.getCurrentState();
            oldMeasurements=currentMeasurements;
            rewardForEp=0;
            for i in range(0,self.numOfSteps):
                currentState = self.getStateFromMeasurements_2([oldMeasurements,currentMeasurements]);
                actionIndex = self.q_table[currentState].idxmax();
                #print(q_table[currentState].unique())
                action = self.getActionFromIndex(actionIndex);
                oldMeasurements=currentMeasurements;
                currentMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1]);
                rewardForEp+=reward;
                #print(self.env_2bus.net.res_bus.vm_pu)
                #print(self.env_2bus.net.res_line)
            rewards.append(rewardForEp);
            #print(sum(rewards))
        plt.scatter(list(range(0, len(rewards))), rewards)
        plt.show();

    def testAllActions(self):
        allRewards=[];
        greedyReward=0;

        self.env_2bus.reset();
        #print(env_2bus.net.load)

        currentMeasurements = self.env_2bus.getCurrentState();
        oldMeasurements=currentMeasurements;
        for j in range(0,self.numOfSteps):
            print('Step: ' + str(j))
            print('Measurements before taking action ')
            print('v: '+str(self.env_2bus.net.res_bus.vm_pu[1])+'; lp deviation:  '+str(np.std(self.env_2bus.net.res_line.loading_percent)))
            #print(self.env_2bus.net.res_line.loading_percent[1])
            #print(self.env_2bus.net.res_line.loading_percent[0])
            #print(np.std(self.env_2bus.net.res_line.loading_percent))
            rewards = [];
            compensation = []
            measurements = []
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
            allRewards.append(rewards)
            currentState = self.getStateFromMeasurements_2([oldMeasurements, currentMeasurements]);
            actionIndex = self.q_table[currentState].idxmax();
            action = self.getActionFromIndex(actionIndex);
            oldMeasurements = currentMeasurements;
            currentMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1]);
            print('algorithm reward: '+str(reward));
            print(compensation[actionIndex])
            print('max possible reward from all actions: '+str(max(rewards)));
            print(compensation[rewards.index(max(rewards))])
            print('Measurements after taking algorithm best action ')
            print(measurements[actionIndex])
            greedyReward+=rewards[actionIndex]
        #action = getActionFromIndex(actionIndex);
        #nextStateMeasurements2, reward2, done2 = env_2bus.takeAction(action[0], action[1]);
        #print(env_2bus.stateIndex-1)
        #print(reward2);
        #print(rewards)
        print('Reward From Greedy Actions for entire episode using algorithm: '+str(greedyReward))

        print('Max Reward Possible by acting greedily at each step:'+str(sum([max(x) for x in allRewards])))

    def compareWith(self,models):
        rewards = [[]];
        for j in range(0, 100):
            self.env_2bus.reset();
            for i in models:
                oldIndex=i.env_2bus.stateIndex;
                i.env_2bus.stateIndex=self.env_2bus.stateIndex;
                i.env_2bus.net.switch.at[0, 'closed'] = False
                i.env_2bus.net.switch.at[1, 'closed'] = True
                i.env_2bus.k_old = 0;
                i.env_2bus.q_old = 0;
                i.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex, oldIndex);
                i.env_2bus.runEnv(False);
                if len(rewards) < len(models)+1:
                    rewards.append([]);


            for k in range(0,len(models)+1):
                currentModel=self if k == len(models) else models[k];
                currentMeasurements = currentModel.env_2bus.getCurrentState();
                oldMeasurements = currentMeasurements;
                rewardForEp = 0;
                for i in range(0, currentModel.numOfSteps):
                    currentState = currentModel.getStateFromMeasurements_2([oldMeasurements, currentMeasurements]);
                    actionIndex = currentModel.q_table[currentState].idxmax();
                    # print(q_table[currentState].unique())
                    action = currentModel.getActionFromIndex(actionIndex);
                    oldMeasurements = currentMeasurements;
                    currentMeasurements, reward, done = currentModel.env_2bus.takeAction(action[0], action[1]);
                    rewardForEp += reward;
                    # print(self.env_2bus.net.res_bus.vm_pu)
                    # print(self.env_2bus.net.res_line)
                rewards[k].append(rewardForEp);
            # print(sum(rewards))
        plt.plot(list(range(0, len(rewards[0]))), rewards[0],color="chocolate")
        #plt.show();
        plt.plot(list(range(0, len(rewards[1]))), rewards[1],  color="green")
        plt.show();


    def train(self):
            print('epsilon: ' + str(self.epsilon))
            print('Has already been  trained for following num of episodes: ' + str(len(self.allRewards)))
            noe=self.numOfEpisodes - len(self.allRewards)
            for i in range(0,noe):
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
                if (i+1) % self.annealAfter == 0:
                    print('Episode: ' + str(len(self.allRewards)) + '; reward:' + str(accumulatedReward))
                    self.epsilon=self.annealingRate*self.epsilon;
                    print('saving checkpoint data')
                    pickledData={'q_table':self.q_table, 'e':self.epsilon, 'allRewards':self.allRewards}
                    pickle.dump(pickledData, open(self.checkPointName, "wb"))

            print('training finished')

    def lp_ref(self):
        return stat.mean(self.env_2bus.net.res_line.loading_percent)

    def runFACTSnoRL(self, v_ref, lp_ref, bus_index_shunt, bus_index_voltage, line_index):
        self.env_2bus.net.switch.at[1, 'closed'] = False
        self.env_2bus.net.switch.at[0, 'closed'] = True
        ##shunt compenstation
        q_comp = self.env_2bus.Shunt_q_comp(v_ref, bus_index_shunt, self.env_2bus.q_old);
        self.env_2bus.q_old = q_comp;
        self.env_2bus.net.shunt.q_mvar = q_comp;
        ##series compensation
        k_x_comp_pu = self.env_2bus.K_x_comp_pu(lp_ref, line_index, self.env_2bus.k_old);
        self.env_2bus.k_old = k_x_comp_pu;
        x_line_pu = self.env_2bus.X_pu(line_index)
        self.env_2bus.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = x_line_pu * k_x_comp_pu
        self.env_2bus.runEnv(runControl=True)
        busVoltage = self.env_2bus.net.res_bus.vm_pu[bus_index_voltage]
        lp_max = max(self.env_2bus.net.res_line.loading_percent)
        return busVoltage, lp_max

    def runFACTSgreedyRL(self, busVoltageIndex, currentState):
        actionIndex = self.q_table[currentState].idxmax()
        action = self.getActionFromIndex(actionIndex)
        nextStateMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1])
        busVoltage = self.env_2bus.net.res_bus.vm_pu[busVoltageIndex]
        lp_max = max(self.env_2bus.net.res_line.loading_percent)
        return nextStateMeasurements, busVoltage, lp_max

    def comparePerformance(self, steps, oper_upd_interval, bus_index_shunt, bus_index_voltage, line_index):
        v_noFACTS = []
        lp_max_noFACTS = []
        v_FACTS = []
        lp_max_FACTS = []
        v_RLFACTS = []
        lp_max_RLFACTS = []

        self.env_2bus.reset()
        stateIndex = self.env_2bus.stateIndex
        loadProfile = self.env_2bus.loadProfile
        while stateIndex + steps > len(loadProfile):
            self.env_2bus.reset()  # Reset to get sufficient number of steps left in time series
            stateIndex = self.env_2bus.stateIndex
            loadProfile = self.env_2bus.loadProfile

        # Need seperate copy for each scenario
        qObj_env_noFACTS = copy.deepcopy(self)
        qObj_env_FACTS = copy.deepcopy(self)
        qObj_env_RLFACTS = copy.deepcopy(self)

        # Initialise states, needed for greedy RL
        # loading
        qObj_env_noFACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
        qObj_env_FACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
        qObj_env_RLFACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
        currentMeasurements = qObj_env_RLFACTS.env_2bus.getCurrentState()
        oldMeasurements = currentMeasurements

        # To plot horizontal axis in nose-curve
        loading_arr = loadProfile[stateIndex:stateIndex + steps]

        # Loop through each load
        for i in range(0, steps):
            # no FACTS
            qObj_env_noFACTS.env_2bus.runEnv(runControl=False)  # No control of shunt comp transformer as it should be disabled.
            v_noFACTS.append(qObj_env_noFACTS.env_2bus.net.res_bus.vm_pu[bus_index_voltage])
            lp_max_noFACTS.append(max(qObj_env_noFACTS.env_2bus.net.res_line.loading_percent))

            # FACTS
            v_ref = 1
            if i % oper_upd_interval == 0:
                lp_reference = qObj_env_FACTS.lp_ref()
            #print(i)
            voltage, lp_max = qObj_env_FACTS.runFACTSnoRL(v_ref, lp_reference, bus_index_shunt, bus_index_voltage,
                                           line_index)  # ERROR LINES:
            v_FACTS.append(voltage)
            lp_max_FACTS.append(lp_max)

            # RLFACTS
            currentState = qObj_env_RLFACTS.getStateFromMeasurements_2([oldMeasurements, currentMeasurements])
            oldMeasurements = currentMeasurements
            currentMeasurements, voltage, lp_max = qObj_env_RLFACTS.runFACTSgreedyRL(bus_index_voltage, currentState)  # runpp is done within this function
            v_RLFACTS.append(voltage)
            lp_max_RLFACTS.append(lp_max)

            # Increment state
            stateIndex += 1
            qObj_env_noFACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
            qObj_env_FACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
            qObj_env_RLFACTS.env_2bus.scaleLoadAndPowerValue(stateIndex, stateIndex - 1)
            #print(i)

        # Adjust arrays so indices are overlapping correctly. otherwise the RL will have i+1:th state where rest has i:th state
        loading_arr.pop(0)
        v_noFACTS.pop(0)
        lp_max_noFACTS.pop(0)
        v_FACTS.pop(0)
        lp_max_FACTS.pop(0)
        v_RLFACTS.pop(-1)
        lp_max_RLFACTS.pop(-1)

        # Make plots
        i_list = list(range(1, steps))
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Time series')
        ax1.set_ylabel('Bus Voltage', color=color)
        ax1.plot(i_list, v_noFACTS, color=color)
        ax1.plot(i_list, v_FACTS, color='g')
        ax1.plot(i_list, v_RLFACTS, color='r')
        ax1.legend(['v no facts', 'v facts' , 'v RL facts'], loc=2)
        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Max line loading [%]', color='m')
        ax2.plot(i_list, lp_max_noFACTS, color=color, linestyle = 'dashed')
        ax2.plot(i_list, lp_max_FACTS, color='g',linestyle = 'dashed')
        ax2.plot(i_list, lp_max_RLFACTS, color='r', linestyle = 'dashed')
        ax2.legend(['max lp no facts', 'max lp facts', 'max lp RL facts'], loc=1)
        plt.show()

        # Nosecurve:
