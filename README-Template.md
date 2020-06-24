# Reinforcement Learning for Grid Voltage Stability with FACTS
With increased penetration of renewable energy sources, maintaining equilibrium
between production and consumption in the world’s electrical power systems
(EPS) becomes more and more challenging. Through this project we aim to obtain voltage stability
in the EPS circuits using Reinforcement Learning. We implement q-learning, DQN and TD3 over a 2 bus system with varying load profiles.
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

### Creating a new benchmark
Benchmarks can be obtained by trying out

```
Give an example
```

### Generating Load Profile Data

There is a load bus in the 2 bus system, where the load keeps on changing. To imitate the load fluctuations that we see commonly in the EPS systems, [enlopy package](https://github.com/energy-modelling-toolkit/enlopy) has been used as it exposes several functionalities that can be used in this context to generate load as well as power profile. In **setup.py** file, createLoadProfile() function has been implemented for this sole purpose. Executing this function shall generate load and power profile for one month with values for every 5 minutes and save them in a pickle file. One can tweak this function as per their requirement and rerun the function to get a different load profile.
```
createLoadProfile()
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

