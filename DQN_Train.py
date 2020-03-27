from DQN_Model import ieee2_net,ieee4_net
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from setup import powerGrid_ieee2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, ieeeBusSystem, lr, memorySize, batchSize,  decayRate, numOfEpisodes, stepsPerEpisode, epsilon, annealingConstant, annealAfter):
        self.eval_net, self.target_net = ieee2_net(12,99,0.3), ieee2_net(12,99)
        USE_CUDA = torch.cuda.is_available();
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity=memorySize;
        self.memory = np.zeros((memorySize, 12 * 2 + 2))  # initialize memory
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
        self.target_update_iter=200
        self.env_2bus = powerGrid_ieee2();
        self.actions=['v_ref:'+str(x)+';lp_ref:'+str(y) for x in self.env_2bus.actionSpace['v_ref_pu'] for y in self.env_2bus.actionSpace['lp_ref']]
        self.allRewards=[];
        self.checkPoint = 'DQN_Checkpoints/dqn_lr' + str(lr) +'bs' +str(batchSize)+'ms'+str(memorySize)+'dr' + str(decayRate) + 'noe' + str(
            numOfEpisodes) + 'spe' + str(stepsPerEpisode) + 'e' + str(epsilon) + 'ac' + str(
            annealingConstant) + 'aa' + str(annealAfter) +'.tar';
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

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        #print(len(self.memory))
        #print(transition)
        # if self.memory_counter < self.memory_capacity:
        #     self.memory.append(transition)
        # else:
        self.memory[index]=transition
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
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :12]).cuda())
        b_a = Variable(torch.LongTensor(b_memory[:, 12:12 + 1].astype(int)).cuda())
        b_r = Variable(torch.FloatTensor(b_memory[:, 12 + 1:12 + 2]).cuda())
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -12:]).cuda())
        #b_s.cuda()
        #b_a.cuda()
        #print(b_s.is_cuda)
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_.cuda()).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.decayRate * (q_next.max(1)[0].unsqueeze(1))  # shape (batch, 1)
        #print(q_target.shape)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        print('epsilon: ' + str(self.epsilon))
        print('Has already been  trained for following num of episodes: ' + str(len(self.allRewards)))
        noe = self.numOfEpisodes - len(self.allRewards)
        self.eval_net.train();
        for i in range(0, noe):
            accumulatedReward = 0;
            self.env_2bus.reset();
            currentState=[];
            for j in range(0,3):
                m=self.env_2bus.getCurrentStateForDQN();
                currentState.extend(m);
                self.env_2bus.stateIndex+=1;
                self.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex,self.env_2bus.stateIndex-1)
                self.env_2bus.runEnv(False);
            currentState.extend(self.env_2bus.getCurrentStateForDQN())
            #print(len(currentState));
            for j in range(0, self.numOfSteps):
                epsComp = np.random.random();
                #currentState = self.getStateFromMeasurements_2([oldMeasurements, currentMeasurements]);
                if epsComp <= self.epsilon:
                    # Exploration Part
                    actionIndex = np.random.choice(99, 1)[0]
                else:
                    # Greedy Approach
                    q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.FloatTensor(currentState),0)).cuda());
                    #print(torch.max(q_value, 1)[1].shape)
                    actionIndex = torch.max(q_value, 1)[1].data.cpu().numpy()[0]  # return the argmax
                    #actionIndex = self.q_table[currentState].idxmax();
                action = self.getActionFromIndex(actionIndex);
                #oldMeasurements = currentMeasurements;
                currentMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1],'dqn');
                oldState=currentState;
                currentState.extend(currentMeasurements);
                currentState.pop(0);
                currentState.pop(1);
                currentState.pop(2);
                #print(len(currentState))
                self.store_transition(oldState, actionIndex, reward, currentState)
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

    def test(self, episodes, numOfStepsPerEpisode):
        rewards=[]
        count=0;
        ul=self.numOfSteps;
        self.eval_net.eval();
        for j in range(0,episodes):
            self.env_2bus.reset();
            currentState = [];
            for j in range(0, 3):
                m = self.env_2bus.getCurrentStateForDQN();
                currentState.extend(m);
                self.env_2bus.stateIndex += 1;
                self.env_2bus.scaleLoadAndPowerValue(self.env_2bus.stateIndex, self.env_2bus.stateIndex - 1)
                self.env_2bus.runEnv(False);
            currentState.extend(self.env_2bus.getCurrentStateForDQN())
            rewardForEp=[];
            for i in range(0,numOfStepsPerEpisode):
                q_value = self.eval_net.forward(Variable(torch.unsqueeze(torch.FloatTensor(currentState), 0)).cuda());
                # print(torch.max(q_value, 1)[1].shape)
                actionIndex = torch.max(q_value, 1)[1].data.cpu().numpy()[0]  # return the argmax
                action = self.getActionFromIndex(actionIndex);
                # oldMeasurements = currentMeasurements;
                currentMeasurements, reward, done = self.env_2bus.takeAction(action[0], action[1], 'dqn');
                #oldState = currentState;
                currentState.extend(currentMeasurements);
                currentState.pop(0);
                currentState.pop(1);
                currentState.pop(2);
                # if i == ul-1:
                #     oldMeasurements = currentMeasurements;
                #     ul+=self.numOfSteps;
                rewardForEp.append(reward);
                #print(self.env_2bus.net.res_bus.vm_pu)
                #print(self.env_2bus.net.res_line)
            rewards.append(sum(rewardForEp));
            #print(sum(rewards))
        plt.scatter(list(range(0, len(rewards))), rewards)
        plt.show();


#print(USE_CUDA)
dqn=DQN(2,0.001,3000,64,0.7,100000,12,1,0.98,800)
dqn.train()