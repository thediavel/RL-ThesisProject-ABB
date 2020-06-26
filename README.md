# Reinforcement Learning for Grid Voltage Stability with FACTS
With increased penetration of renewable energy sources, maintaining equilibrium
between production and consumption in the world’s electrical power systems
(EPS) becomes more and more challenging. Through this project we aim to obtain voltage stability
in the EPS circuits using Reinforcement Learning. We implement q-learning, DQN and TD3 over a 2 bus system with a varying load profile.
The project was carried out at ABB by Vishnu Sharma and Joakim Oldeen as a master thesis project. More information
 can be found here : *Link to published report with same name as project*

## Getting Started

Use the statement below to retrieve the latest version of the project 
  ```
git clone https://github.com/thediavel/RL-ThesisProject-ABB.git
```
### Prerequisites

Project requires python 3.6, torch 1.4.0 and other packages mentioned in **requirements.txt** file.


### Installing
You can start by creating a virtual environment (on Windows)
```
pip install virtualenv
virtualenv venv
.\venv\Scripts\activate
```
Next, you can install all these dependencies using the pip command as shown below

```
pip install setuptools==41.0.0
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## How to create an RL agent
Once all the dependencies have been installed, we can begin using different RL agents and see how they perform.
Start off by importing the needed classes:
```
from QLearning import qLearning
from DQN import DQN
from TD3 import TD3
```

Then you can construct the different RL agents through

```
qlAgent=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
dqnAgent=DQN(ieeeBusSystem=2, lr=0.001, memorySize=2000, batchSize=64, decayRate=0.7, numOfEpisodes=10000, stepsPerEpisode=12, epsilon=1, annealingConstant=0.98, annealAfter=100, targetUpdateAfter=200, expandActions=True, ddqnMode=False)
td3Agent=TD3(ieeeBusSystem=2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)
```
More information about the hyper-parameters are described in the published report linked to above.


If an RL agent with the exact chosen parameters has been fully trained already, the trained agent will be loaded from a saved pickle file and no further training is needed.
If you want to train a new RL agent (or an agent not trained for the full number of episodes set in parameters) you construct an RL agent with a new set of hyper-parameters and then train it through the method train():
```
qlAgent.train()
dqnAgent.train()
td3Agent.train()
```
During training the agent will save checkpoints at certain intervals and do a print() showing the checkpoints have been reached. Furthermore, we log the training results in terms of rewards and training error using tensorboard while training.
Training an agent with the hyper-parameters shown above takes about 5 hours on a device with 8th gen i7 2.6 GHz processor and Nvidia Quadpro P1000 CUDA-enabled graphics card.


## Evaluating performance of an agent
You can use the comparePerformance method to analyse how the algorithms perform on a common test set
```
qlAgent.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
dqnAgent.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
td3Agent.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
```
The comparePerformance method creates a set of graphs for analysing the performance of the RL agent with respect to reward and voltage stability.
Example of printed graphs are Figure 4.2-4.7 in the published report.
There are also a set of printouts which are further explained in the published report.
Running the comparePerformance method takes about 5-10 minutes.

### Creating a new benchmark
The benchmark is a case that is compared to when running the comparePerformance method. The benchmark is further explained in the published report.
If the 2-bus test network is changed, it might call for a new benchmark.

The method for creating a benchmark is located in the DQN class and to be able to use it one must first construct a DQN object, although the benchmark is not an actual RL agent.
To create a new benchmark you first have to make sure there are no existing benchmark files in the 'Data' folder. Once they are removed you can create a new benchmark through
```
dqnAgent.runBenchmarkCase(nTimeSteps=600):
```

Creating a benchmark with 600 time steps is computationally expensive. It took about 70 hours on the same system as the training took 5 hours.
Therefore you might want to divide the creation of the benchmark into intervals. 
This can be done by for first making a benchmark with for example nTimeSteps=50. When that is complete you go into the code 
of the runBenchmarkCase method and change self.env_2bus.stateIndex such that the first lines of code look like this:

```
# Make sure to be in test mode:
self.env_2bus.setMode('test')
self.env_2bus.reset()
self.env_2bus.stateIndex += 3+50 # Plus 3 to be on even state index as RL which uses first 3 time steps as history
````

This way you adjust the starting point 50 time steps, since you already trained for that.
Then you can run runBenchmarkCase(nTimeSteps=50) another time. Then you add another 50 to the self.env_2bus.stateIndex and repeat the process until finished.

### Generating Load Profile Data

There is a load bus in the 2 bus system, where the load keeps on changing. To imitate the load fluctuations that we see commonly in the EPS systems, [enlopy package](https://enlopy.readthedocs.io/en/latest/) has been used as it exposes several functionalities that can be used in this context to generate load as well as power profile. In **setup.py** file, createLoadProfile() function has been implemented for this sole purpose. Executing this function shall generate load and power profile with data for every 5 minutes for 30 days and save them in a pickle file. One can tweak this function as per their requirement and rerun the function to get a different load profile.
```
createLoadProfile()
```

## Built With

* [Pytorch](https://pytorch.org/) - Used to construct and train Neural Networks
* [Pandapower](https://www.pandapower.org/) - Electrical Power Systems Package
* [Enlopy](https://enlopy.readthedocs.io/en/latest/) - Used to generate Load Profile


## Authors

* **Joakim Oldeen*  - [jplOldeen](https://github.com/jplOldeen)
* **Vishnu Sharma*  - [thediavel](https://github.com/thediavel)

## Acknowledgments

This master thesis project has been carried out at and on
behalf of business unit grid integration at ABB FACTS in Västerås. We want
to thank ABB and everyone who has been involved in our process, helping us
investigate the world of reinforcement learning and flexible alternating current
transmission systems.

A special thanks to our supervisor Magnus Tarle for his guidance and input
in the various topics spanning this thesis. Thank you Mats Larsson, for
educating us and helping us find a path. Thank you Jonathan Hanning and
Lexuan Meng for being there when we had questions and for supplying us
with the tools needed to complete our tasks. Also, without the support from
our subject reader Per Mattson, this thesis would have become much harder,
thank you!

Lastly, we would like to thank one another; without our combined background,
knowledge and perseverance this thesis would not have been the same.

Regards

Joakim Oldeen & Vishnu Sharma

