# Reinforcement Learning for Grid Voltage Stability with FACTS
With increased penetration of renewable energy sources, maintaining equilibrium
between production and consumption in the world’s electrical power systems
(EPS) becomes more and more challenging. Through this project we aim to obtain voltage stability
in the EPS circuits using Reinforcement Learning. We implement q-learning, DQN and TD3 over a 2 bus system.
The project was carried out at ABB by Vishnu Sharma and Joakim Oldeen as a master thesis project. More information
 can be found here :

## Getting Started

Use the statement below to retreive the latest version of the project 
  ```
git clone https://github.com/thediavel/RL-ThesisProject-ABB.git
```
### Prerequisites

Project requires python 3.6, torch 1.4.0 and other packages mentioned in **requirements.txt** file.


### Installing

One can install all these dependencies using the pip command as shown below

```
pip install setuptools==41.0.0
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## How to use the RL Agent
Once all the dependencies have been installed, we can begin the using different RL agents and see how they perform
```
from QLearning import qLearning
from DQN import DQN
from TD3 import TD3
```
```
qlAgent=qLearning(learningRate=0.001,decayRate=0.1,numOfEpisodes=48000,stepsPerEpisode=6,epsilon=1,annealingConstant=0.98,annealAfter=400)
dqnAgent=DQN(2, 0.001, 2000, 64, 0.7, 10000, 12, 1, 0.98, 100, 200, True)
td3Agent=TD3(2, lr=0.001, memorySize=2000, batchSize=64,  decayRate=0.1, numOfEpisodes=50000, stepsPerEpisode=12,tau=0.005,policy_freq=2,updateAfter=2)
```

## Running the tests

We can use the comparePerformance method to analyse how the algorithms perform on a common test set
```
qlres.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
dqnres.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
td3res.comparePerformance(steps=600, oper_upd_interval=6, bus_index_shunt=1, bus_index_voltage=1, line_index=1, benchmarkFlag=True)
```
### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
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

