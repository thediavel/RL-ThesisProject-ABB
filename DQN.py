from memory import BasicBuffer
from DQN_Model import ieee2_net,ieee4_net
import torch.nn as nn
from torch.autograd import Variable
from setup import powerGrid_ieee2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import copy
import statistics as stat
from torch.utils.tensorboard import SummaryWriter
import pickle
import math


class DQN:
    def __init__(self, ieeeBusSystem, lr, memorySize, batchSize,  decayRate, numOfEpisodes, stepsPerEpisode, epsilon, annealingConstant, annealAfter, targetUpdateAfter,expandActions=False,ddqnMode=False):
        prefix='ddqn' if ddqnMode else 'dqn'
        self.env_2bus = powerGrid_ieee2('ddqn' if ddqnMode else 'dqn');
        self.ddqnMode=ddqnMode;
        if expandActions:
            self.actions = ['v_ref:' + str(x) + ';lp_ref:' + str(y) for x in self.env_2bus.deepActionSpace['v_ref_pu']
                            for y
                            in self.env_2bus.deepActionSpace['lp_ref']]
        else:
            self.actions = ['v_ref:' + str(x) + ';lp_ref:' + str(y) for x in self.env_2bus.actionSpace['v_ref_pu'] for y
                            in self.env_2bus.actionSpace['lp_ref']]

        op=len(self.actions)

        self.eval_net, self.target_net = ieee2_net(24,op,0.3), ieee2_net(24,op)
        USE_CUDA = torch.cuda.is_available();
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity=memorySize;
        #self.memory = np.zeros((memorySize, 3 * 2 + 3))  # initialize memory
        self.memory=BasicBuffer(self.memory_capacity);
        self.learningRate=lr
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.batch_size=batchSize
        self.numOfEpisodes = numOfEpisodes
        self.annealingRate = annealingConstant
        self.numOfSteps = stepsPerEpisode
        #self.learningRate = learningRate
        self.epsilon=epsilon
        self.decayRate = decayRate
        self.annealAfter = annealAfter
        self.target_update_iter=targetUpdateAfter
        self.allRewards=[];
        self.fileName=prefix+'_lr' + str(lr) +'tua'+str(targetUpdateAfter)+'bs' +str(batchSize)+'ms'+str(memorySize)+'dr' + str(decayRate) + 'noe' + str(
            numOfEpisodes) + 'spe' + str(stepsPerEpisode) + 'e' + str(epsilon) + 'ac' + str(
            annealingConstant) + 'aa' + str(annealAfter)+'op'+str(op);
        self.checkPoint = 'DQN_Checkpoints/'+self.fileName+'.tar';
        print(self.checkPoint)
        if os.path.isfile(self.checkPoint):
            print('loading state values from last saved checkpoint');
            checkpoint = torch.load(self.checkPoint);
            self.eval_net.load_state_dict(checkpoint['evalNet_state_dict'])
            self.target_net.load_state_dict(checkpoint['targetNet_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon=checkpoint['epsilon']
            self.allRewards=checkpoint['allRewards']
            self.learn_step_counter=checkpoint['learn_step_counter']
            self.memory=checkpoint['memory']
            self.memory_counter=checkpoint['memory_counter']

        if USE_CUDA:
            print('GPU Exists')
            self.eval_net.cuda()
            self.target_net.cuda()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            #if next(self.eval_net.parameters()).is_cuda:
            #    print('done')

    def store_transition(self, s, a, r, done,s_):
        #print((s, a, r, s_, done))
        self.memory.push(s, a, r, s_, done)
        #transition = np.hstack((s, [a, r, done], s_))
        # replace the old memory with new memory
        #index = self.memory_counter % self.memory_capacity
        #print(len(self.memory))
        #print(transition)
        # if self.memory_counter < self.memory_capacity:
        #     self.memory.append(transition)
        # else:
        #self.memory[index]=transition
        self.memory_counter += 1

    def getActionFromIndex(self, ind):
        actionString=self.actions[ind];
        actionStringSplitted=actionString.split(';');
        voltage = actionStringSplitted[0].split(':')[1];
        loadingPercent = actionStringSplitted[1].split(':')[1];
        return((int(loadingPercent),float(voltage) ));

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        #sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_s,b_a,b_r,b_s_,dones = self.memory.sample(self.batch_size)
        #print(b_a)
        #b_s.astype(float)
        b_s=Variable(torch.FloatTensor(b_s).cuda())
        b_a = Variable(torch.LongTensor(b_a).cuda())
        #b_r = b_memory[2]
        b_s_ = Variable(torch.FloatTensor(b_s_).cuda())
        #dones = b_memory[4]

        # b_s = Variable(torch.FloatTensor(b_memory[:, :3]).cuda())
        # b_a = Variable(torch.LongTensor(b_memory[:, 3:3 + 1].astype(int)).cuda())
        # b_r = b_memory[:, 3 + 1:3 + 2]
        # dones=Variable(torch.FloatTensor(b_memory[:, 3 + 2:3 + 3]).cuda())
        # b_s_ = Variable(torch.FloatTensor(b_memory[:, -3:]).cuda())
        #print(b_a.shape)
        q_eval = self.eval_net(b_s.unsqueeze(1)).gather(1, b_a.unsqueeze(1))  # shape (batch, 1)
        if self.ddqnMode:
            q_next_eval = self.eval_net(b_s_.unsqueeze(1)).detach()
            q_next_target = self.target_net(b_s_.unsqueeze(1)).detach()
        else:
            q_next = self.target_net(b_s_.unsqueeze(1)).detach()  # detach from graph, don't backpropagate
           # q_target = b_r + self.decayRate * (q_next.max(1)[0].unsqueeze(1))  # shape (batch, 1)

        target = [];
        for i in range(0, self.batch_size):
            terminal = dones[i]
            if terminal:
                target.append(0)
            else:
                if self.ddqnMode:
                    values, action = q_next_eval[i].max(0)
                    #action = np.argmax(q_next_eval[i].)
                    target.append(self.decayRate*q_next_target[i][action].item())
                else:
                    #print(q_next[i])
                    #print(q_next[i].max(0)[0].item())
                    target.append(self.decayRate*q_next[i].max(0)[0].item())
        q_target = np.add(b_r , target)
        #q_target = np.array(q_target)
        q_target = Variable(torch.FloatTensor(q_target).cuda())
        #print(q_target.shape)
        #print(q_target.shape)
        #print(q_eval)
        #print(q_target)
        loss = self.loss_func(q_eval, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.item())
        self.runningLoss+=loss.item()
        self.runningRewards+=sum(b_r)/self.batch_size
        #print(self.learn_step_counter)
        if self.learn_step_counter % 200 == 0:  # every 200 mini-batches...

            # ...log the running loss
            self.writer.add_scalar('training loss',
                              self.runningLoss / 200,
                              self.learn_step_counter)

            self.writer.add_scalar('avg Reward',
                                   self.runningRewards/200,
                                   self.learn_step_counter)
            self.runningRewards = 0;


            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            #self.writer.add_figure('predictions vs. actuals',
            #                  plot_classes_preds(net, inputs, labels),
            #                  global_step=epoch * len(trainloader) + i)
            self.runningLoss = 0.0

    def train(self):
        self.writer = SummaryWriter(
            'runs/' + self.fileName);
        print('epsilon: ' + str(self.epsilon))
        print('Has already been  trained for following num of episodes: ' + str(len(self.allRewards)))
        noe = self.numOfEpisodes - len(self.allRewards)
        self.eval_net.train();
        self.env_2bus.setMode('train')
        self.runningLoss=0;
        self.runningRewards=0;
        for i in range(0, noe):
            accumulatedReward = 0;
            self.env_2bus.reset();
            currentState = [];
            for j in range(0, 3):
                m = self.env_2bus.getCurrentStateForDQN();
                m.extend(m)
                currentState.append(m);
                self.env_2bus.stateIndex += 1;
                self.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex)
                self.env_2bus.runEnv(False);

            currentState.append(self.env_2bus.getCurrentStateForDQN())
            currentState[3].extend(self.env_2bus.getCurrentStateForDQN())
            currentState=np.array(currentState)
            #currentState=self.env_2bus.getCurrentStateForDQN();
            #print(currentState);
            for j in range(0, self.numOfSteps):
                #print(j)
                epsComp = np.random.random();
                #currentState = self.getStateFromMeasurements_2([oldMeasurements, currentMeasurements]);
                if epsComp <= self.epsilon:
                    # Exploration Part
                    actionIndex = np.random.choice(len(self.actions), 1)[0]
                else:
                    # Greedy Approach
                    q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(currentState),0),0)).cuda());
                    #print(torch.max(q_value, 1)[1].shape)
                    actionIndex = torch.max(q_value, 1)[1].data.cpu().numpy()[0]  # return the argmax
                    #actionIndex = self.q_table[currentState].idxmax();
                action = self.getActionFromIndex(actionIndex);
                #oldMeasurements = currentMeasurements;
                currentMeasurements, reward, done, _ = self.env_2bus.takeAction(action[0], action[1]);
                oldState=currentState;
                #currentMeasurements=np.array(currentMeasurements);
                #print(currentMeasurements)
                #np.append(currentState,currentMeasurements,0)
                currentState=np.append(currentState, [currentMeasurements],axis=0)
                #currentState.append(currentMeasurements);
                currentState=np.delete(currentState,0,axis=0);
                #currentState=currentMeasurements;
                #print(currentState)
                self.store_transition(oldState, actionIndex, reward, done ,currentState)
                accumulatedReward += reward;
                if self.memory_counter > self.memory_capacity:
                    self.learn()
                if done:
                    break;

            self.allRewards.append(accumulatedReward);
            if (i + 1) % self.annealAfter == 0:
                print('Episode: ' + str(len(self.allRewards)) + '; reward:' + str(accumulatedReward))
                self.epsilon = self.annealingRate * self.epsilon;
                # if self.learn_step_counter % 200 == 0:
                print('saving checkpoint data')
                torch.save({
                    'epsilon': self.epsilon,
                    'allRewards': self.allRewards,
                    'evalNet_state_dict': self.eval_net.state_dict(),
                    'targetNet_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'learn_step_counter': self.learn_step_counter,
                    'memory': self.memory,
                    'memory_counter': self.memory_counter
                }, self.checkPoint)
        print('training finished')

    def test(self, episodes, numOfStepsPerEpisode, busVoltageIndex,testAllActions):
        rewards=[]
        regrets=[]
        count=0;
        ul=self.numOfSteps;
        self.eval_net.eval();
        voltage=[]
        voltage2 = []
        self.env_2bus.setMode('test')
        if testAllActions:
            copyNetwork = copy.deepcopy(self)

        for j in range(0,episodes):
            self.env_2bus.reset();
            #currentState = self.env_2bus.getCurrentStateForDQN();
            currentState = [];
            for j in range(0, 3):
                m = self.env_2bus.getCurrentStateForDQN();
                m.extend(m)
                currentState.append(m);
                self.env_2bus.stateIndex += 1;
                self.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex)
                self.env_2bus.runEnv(False);

            currentState.append(self.env_2bus.getCurrentStateForDQN())
            currentState[3].extend(self.env_2bus.getCurrentStateForDQN())
            currentState = np.array(currentState)
            rewardForEp=[];
            rew_aa_ForEp=[];
            for i in range(0,numOfStepsPerEpisode):
                q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.FloatTensor(currentState.flatten()), 0)).cuda());
                # print(torch.max(q_value, 1)[1].shape)
                actionIndex = torch.max(q_value, 1)[1].data.cpu().numpy()[0]  # return the argmax
                action = self.getActionFromIndex(actionIndex);
                # oldMeasurements = currentMeasurements;
                currentMeasurements, reward, done, _ = self.env_2bus.takeAction(action[0], action[1]);
                currentState = np.append(currentState, [currentMeasurements], axis=0)
                voltage.append(0.7*currentMeasurements[3] + 0.3*currentMeasurements[0])
                voltage2.append(currentMeasurements[0])
                # currentState.append(currentMeasurements);
                currentState = np.delete(currentState, 0, axis=0);
                rewardForEp.append(reward);
                if testAllActions:
                    _,_,_,_, rew_aa = copyNetwork.runFACTSallActionsRL(busVoltageIndex)
                    rew_aa_ForEp.append(rew_aa)
                if done == True:
                    break;
                #print(self.env_2bus.net.res_bus.vm_pu)
                #print(self.env_2bus.net.res_line)
            rewards.append(sum(rewardForEp));
            if testAllActions:
                regrets.append(sum(rew_aa_ForEp) - sum(rewardForEp))

        # PLot reward and regret
        if testAllActions:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.scatter(list(range(0, len(rewards))), rewards)
        ax1.set_ylabel('Reward')
        ax1.set_xlabel('Episode')
        #print( voltage)
        ax2.plot(list(range(0, len(voltage))), voltage)
        ax2.set_ylabel('voltage')
        ax2.set_xlabel('Episode')

        ax3.plot(list(range(0, len(voltage2))), voltage2)
        ax3.set_ylabel('voltage2')
        ax3.set_xlabel('Episode')
        if testAllActions:
            ax4.scatter(list(range(0, len(regrets))), regrets)
            ax4.set_ylabel('Regret')
            ax4.set_xlabel('Episode')
        plt.show()
        #print(sum(rewards))
        #self.writer.add_graph(self.eval_net, Variable(torch.unsqueeze(torch.FloatTensor(currentState), 0)).cuda())
        #plt.scatter(list(range(0, len(rewards))), rewards)
        #plt.show();

    def lp_ref(self):
        return stat.mean(self.env_2bus.net.res_line.loading_percent)

    def runFACTSnoRL(self, v_ref, lp_ref, bus_index_shunt, bus_index_voltage, line_index, series_comp_enabl):
        # Enable/Disable devices
        self.env_2bus.net.switch.at[1, 'closed'] = False if series_comp_enabl else True
        self.env_2bus.net.switch.at[0, 'closed'] = True
        self.env_2bus.net.controller.in_service[1] = True if series_comp_enabl else False

        # Set reference values
        self.env_2bus.shuntControl.ref = v_ref;
        self.env_2bus.seriesControl.ref = lp_ref;
        self.env_2bus.runEnv(runControl=True)
        busVoltage = self.env_2bus.net.res_bus.vm_pu[bus_index_voltage]
        lp_max = max(self.env_2bus.net.res_line.loading_percent)
        lp_std = np.std(self.env_2bus.net.res_line.loading_percent)
        return busVoltage, lp_max, lp_std

    ## Run the environment controlled by greedy RL
    def runFACTSgreedyRL(self, busVoltageIndex, currentState,takeLastAction):
        #q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.FloatTensor(currentState), 0)).cuda());
        q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.FloatTensor(currentState.flatten()), 0)).cuda())
        actionIndex = torch.max(q_value, 1)[1].data.cpu().numpy()[0]  # return the argmax
        action = self.getActionFromIndex(actionIndex);
        nextStateMeasurements, reward, done, measAfterAction = self.env_2bus.takeAction(action[0], action[1])
        busVoltage = measAfterAction[0]
        lp_max = measAfterAction[1]
        lp_std = measAfterAction[2]
        return nextStateMeasurements, busVoltage, lp_max, lp_std, reward

    ## Run environment and try all actions and choose highest reward
    def runFACTSallActionsRL(self, busVoltageIndex):
        copyNetwork = copy.deepcopy(self)
        reward = 0
        bestAction = []
        rewArr = []

        #Create action space with high resolution:
        copyNetwork.actionSpace = {'v_ref_pu': [i/1000 for i in range(900, 1101)], 'lp_ref': [i for i in range(0, 151)]}
        copyNetwork.actions = ['v_ref:' + str(x) + ';lp_ref:' + str(y) for x in copyNetwork.actionSpace['v_ref_pu']
                        for y in copyNetwork.actionSpace['lp_ref']]

        # Test all actions
        for i in range(0, len(copyNetwork.actions)):
            action = copyNetwork.getActionFromIndex(i)
            nextStateMeas, rew, done, _ = copyNetwork.env_2bus.takeAction(action[0], action[1] )
            copyNetwork.env_2bus.stateIndex -= 1 # increment back as takeAction() increments +1
            rewArr.append(rew)
            if rew > reward:
                bestAction = action   # Save best action
                reward = rew
        # Take best action in actual environment
        currentStateMeasurements, reward, done, measAfterAction = self.env_2bus.takeAction(bestAction[0], bestAction[1])
        busVoltage = measAfterAction[0]
        lp_max = measAfterAction[1]
        lp_std = measAfterAction[2]
        print(' max-min rewArr: ', max(rewArr), min(rewArr))
        return currentStateMeasurements, busVoltage, lp_max, lp_std, reward

    def comparePerformance(self, steps, oper_upd_interval, bus_index_shunt, bus_index_voltage, line_index,benchmarkFlag):
        v_noFACTS = []
        lp_max_noFACTS = []
        lp_std_noFACTS = []
        v_FACTS = []
        lp_max_FACTS = []
        lp_std_FACTS = []
        v_RLFACTS = []
        lp_max_RLFACTS = []
        lp_std_RLFACTS = []
        v_FACTS_noSeries = []
        lp_max_FACTS_noSeries = []
        lp_std_FACTS_noSeries = []
        v_FACTS_eachTS = []
        lp_max_FACTS_eachTS = []
        lp_std_FACTS_eachTS = []
        rewardNoFacts = []
        rewardFacts = []
        rewardFactsEachTS = []
        rewardFactsNoSeries = []
        rewardFactsRL = []
        v_RLFACTS_AfterLoadChange = []
        lp_max_RLFACTS_AfterLoadChange = []
        self.env_2bus.setMode('test')
        self.env_2bus.reset()
        stateIndex = self.env_2bus.stateIndex
        loadProfile = self.env_2bus.loadProfile
        performance=0
        while stateIndex + steps+4 > len(loadProfile):
            self.env_2bus.reset()  # Reset to get sufficient number of steps left in time series
            stateIndex = self.env_2bus.stateIndex
            loadProfile = self.env_2bus.loadProfile

        # Create copy of network for historic measurements
        temp = copy.deepcopy(self)
        #temp.eval_net.eval()

        #Run for history measurements to get full state repr for RL.
        currentState = [];
        for j in range(0, 3):
            m = temp.env_2bus.getCurrentStateForDQN();
            m.extend(m) # creates the 2nd part of the state, "after action" but was no action with RL disabled in this part.
            currentState.append(m);
            temp.env_2bus.stateIndex += 1;
            temp.env_2bus.scaleLoadAndPowerValue(temp.env_2bus.stateIndex)
            temp.env_2bus.runEnv(False);
        currentState.append(self.env_2bus.getCurrentStateForDQN())
        currentState[3].extend(self.env_2bus.getCurrentStateForDQN())
        currentState = np.array(currentState) # Only used for RLFACTS case

        # Need seperate copy for each scenario
        qObj_env_noFACTS = copy.deepcopy(temp)
        qObj_env_FACTS = copy.deepcopy(temp)
        qObj_env_RLFACTS = copy.deepcopy(temp)
        qObj_env_FACTS_noSeries = copy.deepcopy(temp)
        qObj_env_FACTS_eachTS = copy.deepcopy(temp)


        # Make sure FACTS devices disabled for noFACTS case and no Series for that case
        qObj_env_noFACTS.env_2bus.net.switch.at[0, 'closed'] = False
        qObj_env_noFACTS.env_2bus.net.switch.at[1, 'closed'] = True
        qObj_env_FACTS_noSeries.env_2bus.net.switch.at[1, 'closed'] = True

        # To plot horizontal axis in nose-curve
        load_nom_pu = 2 #the nominal IEEE load in pu
        loading_arr = list(load_nom_pu*(loadProfile[stateIndex:stateIndex + steps] / stat.mean(loadProfile)))

        # Loop through each load
        for i in range(0, steps):
            # no FACTS
            qObj_env_noFACTS.env_2bus.runEnv(runControl=False) #No FACTS, no control
            v_noFACTS.append(qObj_env_noFACTS.env_2bus.net.res_bus.vm_pu[bus_index_voltage])
            lp_max_noFACTS.append(max(qObj_env_noFACTS.env_2bus.net.res_line.loading_percent))
            lp_std_noFACTS.append(np.std(qObj_env_noFACTS.env_2bus.net.res_line.loading_percent))
            rewardNoFacts.append((200+(math.exp(abs(1 - qObj_env_noFACTS.env_2bus.net.res_bus.vm_pu[bus_index_voltage]) * 10) * -20) - np.std(qObj_env_noFACTS.env_2bus.net.res_line.loading_percent))/200)

            # FACTS with both series and shunt
            v_ref = 1
            if i % oper_upd_interval == 0:
                lp_reference = qObj_env_FACTS.lp_ref()
                #print('oper', lp_reference)
            voltage, lp_max, lp_std = qObj_env_FACTS.runFACTSnoRL(v_ref, lp_reference, bus_index_shunt, bus_index_voltage,
                                           line_index, True)  # Series compensation enabled
            v_FACTS.append(voltage)
            lp_max_FACTS.append(lp_max)
            lp_std_FACTS.append(lp_std)
            rewFacts=(200+(math.exp(abs(1 - voltage) * 10) * -20) - lp_std)/200 ;

            # FACTS no Series compensation
            voltage, lp_max, lp_std = qObj_env_FACTS_noSeries.runFACTSnoRL(v_ref, lp_reference, bus_index_shunt, bus_index_voltage,
                                                          line_index, False)  # Series compensation disabled
            v_FACTS_noSeries.append(voltage)
            lp_max_FACTS_noSeries.append(lp_max)
            lp_std_FACTS_noSeries.append(lp_std)
            rewFactsNoSeries=(200+(math.exp(abs(1 - voltage) * 10) * -20) - lp_std)/200 ;

            # FACTS with both series and shunt, with system operator update EACH time step
            lp_reference_eachTS = qObj_env_FACTS_eachTS.lp_ref()
            #print('eachTS', lp_reference_eachTS)
            voltage, lp_max, lp_std = qObj_env_FACTS_eachTS.runFACTSnoRL(v_ref, lp_reference_eachTS, bus_index_shunt, bus_index_voltage,
                                                          line_index, True)  # Series compensation enabled
            v_FACTS_eachTS.append(voltage)
            lp_max_FACTS_eachTS.append(lp_max)
            lp_std_FACTS_eachTS.append(lp_std)
            #rewFactsEachTS=(200+(math.exp(abs(1 - voltage) * 10) * -20) - lp_std)/200 ;

            # RLFACTS
            takeLastAction=False;
            qObj_env_RLFACTS.eval_net.eval();
            currentMeasurements, voltage, lp_max, lp_std,r = qObj_env_RLFACTS.runFACTSgreedyRL(bus_index_voltage, currentState, takeLastAction)  # runpp is done within this function
            currentState = np.append(currentState, [currentMeasurements], axis=0)
            currentState = np.delete(currentState, 0, axis=0);

            v_RLFACTS.append(voltage)
            lp_max_RLFACTS.append(lp_max)
            lp_std_RLFACTS.append(lp_std)
            rewardFactsRL.append(r)          # FACTS with both series and shunt
            v_RLFACTS_AfterLoadChange.append(qObj_env_RLFACTS.env_2bus.net.res_bus.vm_pu[1])
            lp_max_RLFACTS_AfterLoadChange.append(max(qObj_env_RLFACTS.env_2bus.net.res_line.loading_percent))

            # Increment state
            stateIndex += 1
            qObj_env_noFACTS.env_2bus.scaleLoadAndPowerValue(stateIndex) #Only for these, rest are incremented within their respective functions
            qObj_env_FACTS.env_2bus.scaleLoadAndPowerValue(stateIndex)
            qObj_env_FACTS_noSeries.env_2bus.scaleLoadAndPowerValue(stateIndex)
            qObj_env_FACTS_eachTS.env_2bus.scaleLoadAndPowerValue(stateIndex)
            rewFacts = 0.7 * rewFacts + 0.3 * (
                        200 + (math.exp(abs(1 - qObj_env_FACTS.env_2bus.net.res_bus.vm_pu[1]) * 10) * -20) - np.std(
                    qObj_env_FACTS.env_2bus.net.res_line.loading_percent)) / 200
            rewardFacts.append(rewFacts)
            rewFactsNoSeries = 0.7 * rewFactsNoSeries + 0.3 * (200 + (
                        math.exp(abs(1 - qObj_env_FACTS_noSeries.env_2bus.net.res_bus.vm_pu[1]) * 10) * -20) - np.std(
                qObj_env_FACTS_noSeries.env_2bus.net.res_line.loading_percent)) / 200
            rewardFactsNoSeries.append(rewFactsNoSeries)
            rewFactsEachTS = 0.7 * rewFacts + 0.3 * (200 + (
                        math.exp(abs(1 - qObj_env_FACTS_eachTS.env_2bus.net.res_bus.vm_pu[1]) * 10) * -20) - np.std(
                qObj_env_FACTS_eachTS.env_2bus.net.res_line.loading_percent)) / 200
            rewardFactsEachTS.append(rewFactsEachTS)  # FACTS with both series and shunt
            if (rewFacts-r < 0.01) and (rewFactsNoSeries-r < 0.01):
                performance += 1;

        print('RL better than no RL in % wrt to reward: ', (performance / steps)*100)
        print('max reward facts:', np.max(rewardFacts))
        print('max reward facts with RL:', np.max(rewardFactsRL))
        print('max reward facts no series:', np.max(rewardFactsNoSeries))
        print('min reward facts:', np.min(rewardFacts))
        print('min reward facts with RL:', np.min(rewardFactsRL))
        print('min reward facts no series:', np.min(rewardFactsNoSeries))
        print('mean reward facts:', np.mean(rewardFacts))
        print('mean reward facts with RL:', np.mean(rewardFactsRL))
        print('mean reward facts no series:', np.mean(rewardFactsNoSeries))
        print('std reward facts:', np.std(rewardFacts))
        print('std reward facts with RL:', np.std(rewardFactsRL))
        print('std reward facts no series:', np.std(rewardFactsNoSeries))

        #remove last element in voltage after load change to get correct dimensions
        v_RLFACTS_AfterLoadChange.pop()

        # Get benchmark from pickle files:
        with open('Data/voltBenchmark.pkl', 'rb') as pickle_file:
            v_RLFACTS_Benchmark = pickle.load(pickle_file)
            v_RLFACTS_Benchmark = v_RLFACTS_Benchmark[0:steps]
        with open('Data/lpmaxBenchmark.pkl', 'rb') as pickle_file:
            lp_max_RLFACTS_Benchmark = pickle.load(pickle_file)
            lp_max_RLFACTS_Benchmark = lp_max_RLFACTS_Benchmark[0:steps]
        with open('Data/lpstdBenchmark.pkl', 'rb') as pickle_file:
            lp_std_RLFACTS_Benchmark = pickle.load(pickle_file)
            #print(lp_std_RLFACTS_Benchmark)
            lp_std_RLFACTS_Benchmark = lp_std_RLFACTS_Benchmark[0:steps]
            #print(lp_std_RLFACTS_Benchmark)
        with open('Data/rewBenchmark.pkl', 'rb') as pickle_file:
            rewardFactsBenchmark = pickle.load(pickle_file)
            rewardFactsBenchmark = rewardFactsBenchmark[0:steps]

            # Make plots
            i_list = list(range(860, 860 + steps))
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_title('Voltage and line loading standard deviation for test set', fontsize=23)
            ax1.set_xlabel('Time step [-]', fontsize=19)
            ax1.set_ylabel('Bus Voltage [pu]', color=color, fontsize=19)
            ax1.plot(i_list, v_noFACTS, color=color)
            ax1.plot(i_list, v_FACTS, color='g')
            ax1.plot(i_list, v_FACTS_noSeries, color='k')
            ax1.plot(i_list, v_RLFACTS, color='r')
            # ax1.plot(i_list, v_FACTS_eachTS, color= 'c')
            if benchmarkFlag:
                ax1.plot(i_list, v_RLFACTS_Benchmark, color='y')

            # ax1.legend(['v no facts', 'v facts' , 'v facts no series comp','v RL facts', 'v RL facts upd each ts', 'v RL benchmark'], loc=2)
            ax1.legend(['v no FACTS', 'v shunt+series', 'v hunt only', 'v DQN', 'v RL benchmark'], loc=2, fontsize=14)
            ax2 = ax1.twinx()

            color = 'tab:blue'
            ax2.set_ylabel('line loading percentage std [% units]', color='m', fontsize=19)
            ax2.plot(i_list, lp_std_noFACTS, color=color, linestyle='dashed')
            ax2.plot(i_list, lp_std_FACTS, color='g', linestyle='dashed')
            ax2.plot(i_list, lp_std_FACTS_noSeries, color='k', linestyle='dashed')
            ax2.plot(i_list, lp_std_RLFACTS, color='r', linestyle='dashed')
            # ax2.plot(i_list, lp_std_FACTS_eachTS, color='c', linestyle = 'dashed')
            if benchmarkFlag:
                ax2.plot(i_list, lp_std_RLFACTS_Benchmark, color='y', linestyle='dashed')
            # ax2.legend(['std lp no facts', 'std lp facts', 'std lp facts no series comp', 'std lp RL facts', 'std lp facts each ts', 'std lp RL benchmark' ], loc=1)
            ax2.legend(['std lp no FACTS', 'std lp shunt+series', 'std lp shunt only', 'std lp DQN',
                        'std lp RL benchmark'], loc=1, fontsize=14)
            plt.grid()
            plt.show()

            # Plot Rewards
            fig2 = plt.figure()
            color = 'tab:blue'
            plt.plot(i_list, rewardNoFacts, Figure=fig2, color=color)
            plt.plot(i_list, rewardFacts, Figure=fig2, color='g')
            plt.plot(i_list, rewardFactsNoSeries, Figure=fig2, color='k')
            plt.plot(i_list, rewardFactsRL, Figure=fig2, color='r')
            # plt.plot(i_list, rewardFactsEachTS, Figure=fig2, color='c')
            if benchmarkFlag:
                plt.plot(i_list, rewardFactsBenchmark, Figure=fig2, color='y')
            plt.title('Rewards along the test set', fontsize=23)
            plt.xlabel('Time step [-]', Figure=fig2, fontsize=19)
            plt.ylabel('Reward [-]', Figure=fig2, fontsize=19)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.legend(['no FACTS', 'FACTS', 'FACTS no series comp', 'RL FACTS', 'FACTS each ts', 'RL FACTS benchmark.'],
            #           loc=1)
            plt.legend(['no FACTS', 'shunt+series', 'shunt only', 'DQN', 'RL benchmark.'],
                       loc=1, fontsize=14)
            plt.grid()
            plt.show()

            ## Calculate measure for comparing RL and Benchmark wrt reward.
            performanceFactsRL = 0
            performanceFacts = 0
            performanceFactsnoSeries = 0
            PerformanceNoFacts = 0
            for i in range(0, steps):
                performanceFactsRL += math.sqrt((rewardFactsRL[i] - rewardFactsBenchmark[i]) ** 2)
                performanceFacts += math.sqrt((rewardFacts[i] - rewardFactsBenchmark[i]) ** 2)
                performanceFactsnoSeries += math.sqrt((rewardFactsNoSeries[i] - rewardFactsBenchmark[i]) ** 2)
                PerformanceNoFacts += math.sqrt((rewardNoFacts[i] - rewardFactsBenchmark[i]) ** 2)

            print('')
            print('performance FACTS RL: ', performanceFactsRL)
            print('performance FACTS shunt+series: ', performanceFacts)
            print('performance FACTS shunt only: ', performanceFactsnoSeries)
            print('performance no FACTS: ', PerformanceNoFacts)

            # Nosecurve:
            loading_arr_sorted = sorted(loading_arr)

            # Sort the measurements
            v_noFACTS_sorted = [x for _, x in sorted(zip(loading_arr, v_noFACTS))]
            lp_max_noFACTS_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_noFACTS))]
            v_FACTS_sorted = [x for _, x in sorted(zip(loading_arr, v_FACTS))]
            lp_max_FACTS_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_FACTS))]
            v_RLFACTS_sorted = [x for _, x in sorted(zip(loading_arr, v_RLFACTS))]
            lp_max_RLFACTS_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_RLFACTS))]
            v_FACTS_noSeries_sorted = [x for _, x in sorted(zip(loading_arr, v_FACTS_noSeries))]
            lp_max_FACTS_noSeries_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_FACTS_noSeries))]
            v_FACTS_eachTS_sorted = [x for _, x in sorted(zip(loading_arr, v_FACTS_eachTS))]
            lp_max_FACTS_eachTS_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_FACTS_eachTS))]
            if benchmarkFlag:
                v_RLFACTS_Benchmark_sorted = [x for _, x in sorted(zip(loading_arr, v_RLFACTS_Benchmark))]
                lp_max_RLFACTS_Benchmark_sorted = [x for _, x in sorted(zip(loading_arr, lp_max_RLFACTS_Benchmark))]
            v_RLFACTS_AfterLoadChange_sorted = [x for _, x in sorted(zip(loading_arr, v_RLFACTS_AfterLoadChange))]
            lp_max_RLFACTS_AfterLoadChange_sorted = [x for _, x in
                                                     sorted(zip(loading_arr, lp_max_RLFACTS_AfterLoadChange))]
            print('')
            print('maximum loading percentage noFACTS  ', max(lp_max_noFACTS))
            print('maximum loading percentage Shunt+Series  ', max(lp_max_FACTS))
            print('maximum loading percentage shunt only  ', max(lp_max_FACTS_noSeries))
            print('maximum loading percentage after action RL: ', max(lp_max_RLFACTS_Benchmark))
            print('maximum loading percentage after load change RL: ', max(lp_max_RLFACTS_AfterLoadChange))

            # Trim arrays to only include values <= X % loading percentage
            lp_limit_for_noseCurve = 100
            lp_max_noFACTS_sorted_trim = [x for x in lp_max_noFACTS_sorted if x <= lp_limit_for_noseCurve]
            lp_max_FACTS_sorted_trim = [x for x in lp_max_FACTS_sorted if x <= lp_limit_for_noseCurve]
            lp_max_RLFACTS_sorted_trim = [x for x in lp_max_RLFACTS_sorted if x <= lp_limit_for_noseCurve]
            lp_max_FACTS_noSeries_sorted_trim = [x for x in lp_max_FACTS_noSeries_sorted if x <= lp_limit_for_noseCurve]
            lp_max_FACTS_eachTS_sorted_trim = [x for x in lp_max_FACTS_eachTS_sorted if x <= lp_limit_for_noseCurve]
            if benchmarkFlag:
                lp_max_RLFACTS_Benchmark_sorted_trim = [x for x in lp_max_RLFACTS_Benchmark_sorted if
                                                        x <= lp_limit_for_noseCurve]
            lp_max_RLFACTS_AfterLoadChange_sorted_trim = [x for x in lp_max_RLFACTS_AfterLoadChange_sorted if
                                                          x <= lp_limit_for_noseCurve]

            v_noFACTS_sorted_trim = v_noFACTS_sorted[0:len(lp_max_noFACTS_sorted_trim)]
            v_FACTS_sorted_trim = v_FACTS_sorted[0:len(lp_max_FACTS_sorted_trim)]
            v_RLFACTS_sorted_trim = v_RLFACTS_sorted[0:len(lp_max_RLFACTS_sorted_trim)]
            v_FACTS_noSeries_sorted_trim = v_FACTS_noSeries_sorted[0:len(lp_max_FACTS_noSeries_sorted_trim)]
            v_FACTS_eachTS_sorted_trim = v_FACTS_eachTS_sorted[0:len(lp_max_FACTS_eachTS_sorted_trim)]
            if benchmarkFlag:
                v_RLFACTS_Benchmark_sorted_trim = v_RLFACTS_Benchmark_sorted[
                                                  0:len(lp_max_RLFACTS_Benchmark_sorted_trim)]
            v_RLFACTS_AfterLoadChange_sorted_trim = v_RLFACTS_AfterLoadChange_sorted[
                                                    0:len(lp_max_RLFACTS_AfterLoadChange_sorted_trim)]

            loading_arr_plot_noFACTS = loading_arr_sorted[0:len(lp_max_noFACTS_sorted_trim)]
            loading_arr_plot_FACTS = loading_arr_sorted[0:len(lp_max_FACTS_sorted_trim)]
            loading_arr_plot_RLFACTS = loading_arr_sorted[0:len(lp_max_RLFACTS_sorted_trim)]
            loading_arr_plot_FACTS_noSeries = loading_arr_sorted[0:len(lp_max_FACTS_noSeries_sorted_trim)]
            loading_arr_plot_FACTS_eachTS = loading_arr_sorted[0:len(lp_max_FACTS_eachTS_sorted_trim)]
            if benchmarkFlag:
                loading_arr_plot_RLFACTS_Benchmark = loading_arr_sorted[0:len(lp_max_RLFACTS_Benchmark_sorted_trim)]
            loading_arr_plot_RLFACTS_AfterLoadChange = loading_arr_sorted[
                                                       1:len(lp_max_RLFACTS_AfterLoadChange_sorted_trim) + 1]

            # Print result wrt trimmed voltage
            print('')
            print('max voltage facts:', np.max(v_FACTS_sorted_trim))
            print('max voltage facts with RL:', np.max(v_RLFACTS_sorted_trim))
            print('max voltage facts no series:', np.max(v_FACTS_noSeries_sorted_trim))
            print('min voltage facts:', np.min(v_FACTS_sorted_trim))
            print('min voltage facts with RL:', np.min(v_RLFACTS_sorted_trim))
            print('min voltage facts no series:', np.min(v_FACTS_noSeries_sorted_trim))
            print('mean voltage facts:', np.mean(v_FACTS_sorted_trim))
            print('mean voltage facts with RL:', np.mean(v_RLFACTS_sorted_trim))
            print('mean voltage facts no series:', np.mean(v_FACTS_noSeries_sorted_trim))
            print('std voltage facts:', np.std(v_FACTS_sorted_trim))
            print('std voltage facts with RL:', np.std(v_RLFACTS_sorted_trim))
            print('std voltage facts no series:', np.std(v_FACTS_noSeries_sorted_trim))
            print('')
            print('max voltage RL after load change', np.max(v_RLFACTS_AfterLoadChange))
            print('min voltage RL after load change', np.min(v_RLFACTS_AfterLoadChange))
            print('mean voltage RL after load change', np.mean(v_RLFACTS_AfterLoadChange))
            print('std voltage RL after load change', np.std(v_RLFACTS_AfterLoadChange))

            # Plot Nose Curve
            fig3 = plt.figure()
            color = 'tab:blue'
            markersize = 13
            plt.scatter(loading_arr_plot_noFACTS, v_noFACTS_sorted_trim, Figure=fig3, color=color, s=markersize)
            plt.scatter(loading_arr_plot_FACTS, v_FACTS_sorted_trim, Figure=fig3, color='g', s=markersize)
            plt.scatter(loading_arr_plot_FACTS_noSeries, v_FACTS_noSeries_sorted_trim, Figure=fig3, color='k',
                        s=markersize)
            plt.scatter(loading_arr_plot_RLFACTS, v_RLFACTS_sorted_trim, Figure=fig3, color='r', s=markersize)
            plt.scatter(loading_arr_plot_RLFACTS_AfterLoadChange, v_RLFACTS_AfterLoadChange_sorted_trim, Figure=fig3,
                        color='c', s=markersize)
            if benchmarkFlag:
                plt.scatter(loading_arr_plot_RLFACTS_Benchmark, v_RLFACTS_Benchmark_sorted_trim, Figure=fig3, color='y',
                            s=markersize)
            plt.title('Nose-curve from test set with sorted voltage levels', fontsize=23)
            plt.xlabel('Loading [pu]', Figure=fig3, fontsize=19)
            plt.ylabel('Bus Voltage [pu]', Figure=fig3, color=color, fontsize=19)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.legend(['v no FACTS', 'v FACTS', 'v FACTS no series comp','v RL FACTS', 'v FACTS each ts', 'v RL FACTS benchmark.'], loc=1)
            plt.legend(['no FACTS', 'shunt+series', 'shunt only', 'DQN $v_1$', 'DQN $v_2$',
                        'RL benchmark.'], loc=1, fontsize=14)
            plt.grid()
            plt.show()



    # This method replaces runFACTSallActionsRL as this saves result into picke file to be used after it has been run once.
    def runBenchmarkCase(self, nTimeSteps):
        # Make sure to be in test mode:
        self.env_2bus.setMode('test')
        self.env_2bus.reset()
        self.env_2bus.stateIndex += 3+50+100+150+150 # Plus 3 to be on even state index as RL which uses first 3 time steps as history
        self.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex )
        print(len(self.env_2bus.loadProfile))

        # Construct arrays for result
        voltBenchmark = []
        lpstdBenchmark = []
        lpmaxBenchmark = []
        rewBenchmark = []
        voltBenchmarkPrev = []
        lpstdBenchmarkPrev = []
        lpmaxBenchmarkPrev = []
        rewBenchmarkPrev = []


        for k in range(nTimeSteps):
            # Reset reward and best action for each time step
            copyNetwork = copy.deepcopy(self)  # Needed for iterations for each time step. Reset here to help with solver bug

            # Create action space with high resolution:
            copyNetwork.actionSpace = {'v_ref_pu': [i * 5 / 1000 for i in range(180, 221)],
                                       'lp_ref': [i * 2 for i in range(0, 76)]}
            copyNetwork.actions = ['v_ref:' + str(x) + ';lp_ref:' + str(y) for x in copyNetwork.actionSpace['v_ref_pu']
                                   for y in copyNetwork.actionSpace['lp_ref']]

            reward = 0
            bestAction = (0,0)

            # Test all actions
            for i in range(0, len(copyNetwork.actions)):
                # FACTS settings from previous step
                serXtemp = self.env_2bus.net.impedance.loc[0, ['xft_pu']]
                shuntQtemp =self.env_2bus.net.shunt.q_mvar[0]

                #New action
                action = copyNetwork.getActionFromIndex(i)
                indTemp = copyNetwork.env_2bus.stateIndex
                nextStateMeas, rew, done, measAftA = copyNetwork.env_2bus.takeAction(action[0], action[1])
                if indTemp+1 == copyNetwork.env_2bus.stateIndex: # IF stateIndex was incremented in TakeAction
                    copyNetwork.env_2bus.stateIndex -= 1  # increment back as takeAction() increments +1
                    copyNetwork.env_2bus.scaleLoadAndPowerValue(copyNetwork.env_2bus.stateIndex)  # Has to be rescaled for next iteration
                else:
                    self.env_2bus.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = serXtemp
                    self.env_2bus.net.shunt.q_mvar[0] = shuntQtemp
                #print('cn stateIndex', copyNetwork.env_2bus.stateIndex, 'loading: ', copyNetwork.env_2bus.net.load.p_mw[0])
                if rew > reward:
                    bestAction = action  # Save best action
                    reward =  rew
                    # print(reward)
                    # print(action)
                    # print(measAftA[0])
                    # print(measAftA[2])
                    # print(copyNetwork.env_2bus.net.res_bus.vm_pu[1])
                    # print(copyNetwork.env_2bus.net.load.p_mw[0])
                    # print(copyNetwork.env_2bus.net.impedance.loc[0, ['xft_pu']])
                    # print(copyNetwork.env_2bus.net.shunt.q_mvar[0])
                    # print('')

            # Take best action in actual environment
            print(self.env_2bus.net.load.p_mw)
            print('')
            currentStateMeasurements, reward, done, measAfterAction = self.env_2bus.takeAction(bestAction[0], bestAction[1])
            voltBenchmark.append(measAfterAction[0])
            lpmaxBenchmark.append(measAfterAction[1])
            lpstdBenchmark.append(measAfterAction[2])
            rewBenchmark.append(reward)
            # print(' max-min rewArr: ', max(rewArr), min(rewArr))

            # Increment index and loading for copyNetwork for next time step
            #copyNetwork.env_2bus.stateIndex += 1
            #copyNetwork.env_2bus.scaleLoadAndPowerValue(copyNetwork.env_2bus.stateIndex)

            # Print best result for this time step
            print('State index done:', self.env_2bus.stateIndex-1, 'reward: ', reward, 'action: ', bestAction)

        ## Save results into Pickle files
        # Check if file name already exists, extend
        voltFile = 'Data/voltBenchmark.pkl'
        lpstdFile = 'Data/lpstdBenchmark.pkl'
        lpmaxFile = 'Data/lpmaxBenchmark.pkl'
        rewFile = 'Data/rewBenchmark.pkl'

        if os.path.isfile(voltFile):
            with open(voltFile, 'rb') as pickle_file:
                voltBenchmarkPrev = pickle.load(pickle_file)
        if os.path.isfile(lpstdFile):
            with open(lpstdFile, 'rb') as pickle_file:
                lpstdBenchmarkPrev = pickle.load(pickle_file)
        if os.path.isfile(lpmaxFile):
            with open(lpmaxFile, 'rb') as pickle_file:
                lpmaxBenchmarkPrev = pickle.load(pickle_file)
        if os.path.isfile(rewFile):
            with open(rewFile, 'rb') as pickle_file:
                rewBenchmarkPrev = pickle.load(pickle_file)

        voltBenchmarkPrev.extend(voltBenchmark)
        lpstdBenchmarkPrev.extend(lpstdBenchmark)
        lpmaxBenchmarkPrev.extend(lpmaxBenchmark)
        rewBenchmarkPrev.extend(rewBenchmark)
        #print(voltBenchmarkPrev)
        #print(voltBenchmark)
        #print(lpmaxBenchmarkPrev)
        #print(rewBenchmarkPrev)

        pickle.dump(voltBenchmarkPrev, open("Data/voltBenchmark.pkl", "wb"))
        pickle.dump(lpstdBenchmarkPrev, open("Data/lpstdBenchmark.pkl", "wb"))
        pickle.dump(lpmaxBenchmarkPrev, open("Data/lpmaxBenchmark.pkl", "wb"))
        pickle.dump(rewBenchmarkPrev, open("Data/rewBenchmark.pkl", "wb"))



