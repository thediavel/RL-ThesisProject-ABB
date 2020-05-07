import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import BasicBuffer
import os
from torch.utils.tensorboard import SummaryWriter
from setup import powerGrid_ieee2
from torch.autograd import Variable


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,p=0.3):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.drop_layer = nn.Dropout(p=p)
        #self.max_action = Variable(torch.FloatTensor(max_action).cuda())

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.drop_layer(a);
        #print(self.max_action)
        #print(torch.sigmoid(self.l3(a)).flatten())
        #print(torch.mul(self.max_action,torch.sigmoid(self.l3(a)).flatten()))
        return torch.sigmoid(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,p=0.3):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.drop_layer = nn.Dropout(p=p)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.drop_layer(q1);
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.drop_layer(q2);
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3:
    def __init__(self, ieeeBusSystem, lr, memorySize, batchSize,  decayRate, numOfEpisodes, stepsPerEpisode,tau,policy_noise,policy_freq,noise_clip):
        prefix='td3'
        self.env_2bus = powerGrid_ieee2('td3');
        self.max_actions=[150,1.1]
        self.actor = Actor(24, 2)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(24, 2)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        USE_CUDA = torch.cuda.is_available();
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity = memorySize;
        # self.memory = np.zeros((memorySize, 3 * 2 + 3))  # initialize memory
        self.memory = BasicBuffer(self.memory_capacity);
        self.learningRate = lr
        #self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.batch_size = batchSize
        self.numOfEpisodes = numOfEpisodes
        #self.annealingRate = annealingConstant
        self.numOfSteps = stepsPerEpisode
        # self.learningRate = learningRate
        #self.epsilon = epsilon
        self.decayRate = decayRate
        #self.annealAfter = annealAfter
        self.allRewards = [];
        self.fileName = prefix + '_lr' + str(lr) +  'bs' + str(batchSize) + 'ms' + str(
            memorySize) + 'dr' + str(decayRate) + 'noe' + str(
            numOfEpisodes) + 'spe' + str(stepsPerEpisode)+'pf'+str(policy_freq)+'tau'+str(tau);
        self.checkPoint = 'TD3_Checkpoints/' + self.fileName + '.tar';
        print(self.checkPoint)
        if os.path.isfile(self.checkPoint):
            print('loading state values from last saved checkpoint');
            checkpoint = torch.load(self.checkPoint);
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actorTarget_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['criticTarget_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actorOptimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['criticOptimizer_state_dict'])
            #self.epsilon = checkpoint['epsilon']
            self.allRewards = checkpoint['allRewards']
            self.learn_step_counter = checkpoint['learn_step_counter']
            self.memory = checkpoint['memory']
            self.memory_counter = checkpoint['memory_counter']

        if USE_CUDA:
            print('GPU Exists')
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
            for state in self.actor_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            for state in self.critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()



        #self.max_action = max_action
        #self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        return self.actor(state).cpu().data.numpy().flatten()

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

    def train(self):
        self.writer = SummaryWriter(
            'runs/' + self.fileName);
        print('Has already been  trained for following num of episodes: ' + str(len(self.allRewards)))
        noe = self.numOfEpisodes - len(self.allRewards)
        self.actor.train();
        self.critic.train();
        self.env_2bus.setMode('train')
        self.runningCriticLoss = 0;
        self.runningActorLoss = 0;
        self.runningRewards = 0;
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
            currentState = np.array(currentState)
            # currentState=self.env_2bus.getCurrentStateForDQN();
            # print(currentState);
            for j in range(0, self.numOfSteps):
                action=self.select_action(currentState)
                s = np.random.normal(0, 0.1,2)
                #print(s[0])
                action[0]=max(0,min(1,action[0]+s[0]))
                action[1]=max(0,min(1,action[1]+s[1]))

                #print(action)
                currentMeasurements, reward, done, _ = self.env_2bus.takeAction(action[0]*self.max_actions[0], action[1]*self.max_actions[1]);
                oldState = currentState;
                currentState = np.append(currentState, [currentMeasurements], axis=0)
                currentState = np.delete(currentState, 0, axis=0);
                self.store_transition(oldState, action, reward, 0 if done else 1, currentState)
                accumulatedReward += reward;
                if self.memory_counter > self.memory_capacity:
                    self.learn()
                if done:
                    break;

            self.allRewards.append(accumulatedReward);
            if (i + 1) % 200 == 0:
                print('Episode: ' + str(len(self.allRewards)) + '; reward:' + str(accumulatedReward))
                print('saving checkpoint data')
                torch.save({
                    'allRewards': self.allRewards,
                    'actor_state_dict': self.actor.state_dict(),
                    'actorTarget_state_dict': self.actor_target.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'criticTarget_state_dict': self.critic_target.state_dict(),
                    'actorOptimizer_state_dict': self.actor_optimizer.state_dict(),
                    'criticOptimizer_state_dict': self.critic_optimizer.state_dict(),
                    'learn_step_counter': self.learn_step_counter,
                    'memory': self.memory,
                    'memory_counter': self.memory_counter
                }, self.checkPoint)
        print('training finished')

    def learn(self):
        self.learn_step_counter += 1

        # Sample replay buffer
        state, action, reward,next_state, done = self.memory.sample(self.batch_size)
        state = Variable(torch.FloatTensor(state).cuda())
        action = Variable(torch.FloatTensor(action).cuda())
        done = Variable(torch.FloatTensor(done).cuda())
        next_state = Variable(torch.FloatTensor(next_state).cuda())
        reward=Variable(torch.FloatTensor(reward).cuda())
        #print(action.shape)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            #print(torch.randn_like(action))
            #noise = (
            #        torch.randn_like(action) * self.policy_noise
            #).clamp(-self.noise_clip, self.noise_clip)

            n = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.2]))
            noise=n.sample(action.shape).clamp(-self.noise_clip, self.noise_clip).squeeze(2).cuda()
            #print(next_state)
            #print(noise.shape)
            #print(self.actor_target(next_state).shape)
            #print(self.actor_target(next_state))
            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(0, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            #print((target_Q*done.unsqueeze(1)).shape)
            target_Q = reward.unsqueeze(1) + self.decayRate * (target_Q * done.unsqueeze(1))

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        #print(current_Q1.shape)
        #print(target_Q.shape)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.runningCriticLoss += critic_loss.item()
        self.runningRewards += torch.sum(reward).item() / self.batch_size
        # Delayed policy updates
        if self.learn_step_counter % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.runningActorLoss += actor_loss.item()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.learn_step_counter % 50 == 0:  # every 200 mini-batches...

            # ...log the running loss
            self.writer.add_scalar('training loss- Critic',
                              self.runningCriticLoss / 50,
                              self.learn_step_counter)
            self.writer.add_scalar('training loss- Actor',
                                   self.runningActorLoss / 50,
                                   self.learn_step_counter)
            self.writer.add_scalar('avg Reward',
                                   self.runningRewards/50,
                                   self.learn_step_counter)
            self.runningRewards = 0;
            self.runningCriticLoss=0;
            self.runningActorLoss = 0;