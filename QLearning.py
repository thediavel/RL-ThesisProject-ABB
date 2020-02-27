from setup import powerGrid_ieee2
import pandas as pd
import numpy as np
import math

voltageRanges=['<0.75','0.75-0-79','0.8-0-84','0.85-0.89','0.9-0.94','0.95-0.99','1-1.04','1.05-1.09','1.1-1.14','1.15-1.19','1.2-1.24','>=1.25'];
loadingPercentRange=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109','110-119','120-129','130-139','140-149','150 and above'];
states=['v:' + x + ';l:' + y for x in voltageRanges for y in loadingPercentRange]
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

def getActionFromIndex(ind):
    actionString=actions[ind];
    actionStringSplitted=actionString.split(';');
    voltage = actionStringSplitted[0].split(':')[1];
    loadingPercent = actionStringSplitted[1].split(':')[1];
    return((float(voltage), int(loadingPercent)));

q_table = pd.DataFrame(0, index=np.arange(len(actions)), columns=states)
epsilon = 0.7
numOfEpisodes = 1000
numOfSteps = 36
learningRate = 0.01
decayRate = 0.9

for i in range(0,numOfEpisodes):
    accumulatedReward=0;
    env_2bus.reset();
    currentMeasurements = env_2bus.getCurrentState();
    for j in range(0,numOfSteps):
        epsComp = np.random.random();
        currentState=getStateFromMeasurements(currentMeasurements);
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

            break;
        else:
            nextState = getStateFromMeasurements(nextStateMeasurements);






print(getStateFromMeasurements((0.77,123)))
print(getActionFromIndex(56))
#print(q_table)
#createStateSpace()