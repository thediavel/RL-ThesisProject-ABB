from setup import powerGrid_ieee2
import pandas as pd
import numpy as np

voltageRanges=['<0.75','0.75-0-80','0.8-0-85','0.85-0.9','0.9-0.95','0.95-1','1-1.05','1.05-1.1','1.1-1.15','1.15-1.2','1.2-1.25','>1.25'];
LoadingPercentRange=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100-109','110-119','125-129','130-139','140-149','150 and above'];
states=['v:' + x + ';l:' + y for x in voltageRanges for y in LoadingPercentRange]
env_2bus=powerGrid_ieee2();
actions=['v_ref:'+str(x)+';lp_ref:'+str(y) for x in env_2bus.actionSpace['v_ref_pu'] for y in env_2bus.actionSpace['lp_ref']]
q_table = pd.DataFrame(10, index=np.arange(len(actions)), columns=states)
print(q_table)
#createStateSpace()