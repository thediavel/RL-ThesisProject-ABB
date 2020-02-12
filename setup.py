import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
import enlopy as el
import numpy as np
import pandas as pd
import pickle
pd.options.display.float_format = '{:.4g}'.format

class powerGrid:
    def __init__(self):
        print('in init. Here we lay down the grid structure and load some random state values based on IEEE dataset');

        with open('JanLoadEvery5mins.pkl', 'rb') as pickle_file:
            self.loadProfile = pickle.load(pickle_file)
        with open('generatorValuesEvery5mins.pkl', 'rb') as pickle_file:
            self.powerProfile = pickle.load(pickle_file)

        ## Basic ieee 4bus system
        self.net = pp.networks.case4gs()
        ####Shunt FACTS device (bus 1)
        # MV bus
        bus_SVC = pp.create_bus(self.net, name='MV SVCtrafo bus', vn_kv=69, type='n', geodata=(-2, 2.5), zone=2,
                                max_vm_pu=1.1,
                                min_vm_pu=0.9)
        # Trafo
        trafoSVC = pp.create_transformer_from_parameters(self.net, hv_bus=1, lv_bus=4, in_service=True,
                                                         name='trafoSVC', sn_mva=110, vn_hv_kv=230, vn_lv_kv=69,
                                                         vk_percent=12, vkr_percent=0.26, pfe_kw=55, i0_percent=0.06,
                                                         shift_degree=0, tap_side='hv', tap_neutral=0, tap_min=-9,
                                                         tap_max=9,
                                                         tap_step_percent=1.5, tap_step_degree=0,
                                                         tap_phase_shifter=False)
        # Breaker between grid HV bus and trafo HV bus to connect buses
        sw_SVC = pp.create_switch(self.net, bus=1, element=0, et='t', type='CB', closed=True)
        # Shunt devices connected with MV bus
        TSC1 = pp.create_shunt(self.net, bus_SVC, -50, in_service=True, name='TSC1', step=1)
        TSC2 = pp.create_shunt(self.net, bus_SVC, -50, in_service=True, name='TSC2', step=1)
        TCR = pp.create_shunt(self.net, bus_SVC, 1, in_service=True, name='TCR', step=80)
        ####Series device (at line 3, in middle between bus 2 and 3)
        # Add intermediate buses for bypass and series compensation impedance
        bus_SC1 = pp.create_bus(self.net, name='SC bus 1', vn_kv=230, type='n', geodata=(-1, 3.1), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        bus_SC2 = pp.create_bus(self.net, name='SC bus 2', vn_kv=230, type='n', geodata=(-1, 3.0), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        sw_SC_bypass = pp.create_switch(self.net, bus=5, element=6, et='b', type='CB', closed=True)
        imp_SC = pp.create_impedance(self.net, from_bus=5, to_bus=6, rft_pu=0.01272, xft_pu=-0.0636,
                                     rtf_pu=0.01272, xtf_pu=-0.0636, sn_mva=250, in_service=True)
        # Adjust orginal Line 3 to connect to new buses isntead.
        self.net.line.at[3, ['length_km', 'to_bus', 'name']] = [0.5, 5, 'line1_SC']
        lineSC2 = pp.create_line_from_parameters(self.net, name='line2_SC',
                                                 c_nf_per_km=self.net.line.at[3, 'c_nf_per_km'],
                                                 df=self.net.line.at[3, 'df'], from_bus=6,
                                                 g_us_per_km=self.net.line.at[3, 'g_us_per_km'],
                                                 in_service=self.net.line.at[3, 'in_service'], length_km=0.5,
                                                 max_i_ka=self.net.line.at[3, 'max_i_ka'],
                                                 max_loading_percent=self.net.line.at[3, 'max_loading_percent'],
                                                 parallel=self.net.line.at[3, 'parallel'],
                                                 r_ohm_per_km=self.net.line.at[3, 'r_ohm_per_km'],
                                                 std_type=self.net.line.at[3, 'std_type'], to_bus=3,
                                                 type=self.net.line.at[3, 'type'],
                                                 x_ohm_per_km=self.net.line.at[3, 'x_ohm_per_km']);

    def takeAction(self, action):
        print('take action and observer the grid for reactance component or power losses');

    def reset(self):
        print('reset the current environment for next episode');
        for i in range(0, self.net.load['bus'].count()):
            print(self.net.load.iloc[i]['p_mw']);

    def plotGridFlow(self):
        print('plotting powerflow for the current state')
        plot.simple_plot(self.net)


#env=powerGrid();
#env.plotGridFlow();
#env.reset();

##Load Profile data has been pickled already, do not run this function for now
def createLoadProfile():
    ML  = (np.cos(2 * np.pi/12 * np.linspace(0,11,12)) * 50 + 100 ) * 1000  # monthly load
    ML = el.make_timeseries(ML) #convenience wrapper around pd.DataFrame with pd.DateTimeindex
    #print(ML)
    DWL =  el.gen_daily_stoch_el() #daily load working
    DNWL = el.gen_daily_stoch_el() #daily load non working
    #print(sum(DNWL))
    Weight = .60 # i.e energy will be split 55% in working day 45% non working day

    Load1 =  el.gen_load_from_daily_monthly(ML, DWL, DNWL, Weight)
    Load1.name = 'L1'
    Load1=Load1.round();
    #print(Load1)

    disag_profile = np.random.rand(60)
    JanLoadEveryMinute=el.generate.disag_upsample(Load1[0:744],disag_profile, to_offset='min');
    JanLoadEvery5mins=[];
    l=0;
    for i in range(0,JanLoadEveryMinute.shape[0]):
        l=l+JanLoadEveryMinute[i];
        if np.mod(i+1,5) == 0:
            JanLoadEvery5mins.append(l);
            l=0;

    windDataDF = pd.read_excel('WindEnergyData.xlsx');
    generatorValuesEvery5mins=[];
    for i in range(1,windDataDF.shape[0]):
        randomValue=np.random.choice(100, 1)[0]
        if randomValue > windDataDF.iloc[i]['DE_50hertz_wind_generation_actual']:
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
        else :
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] - randomValue)
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] + randomValue)
        generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
    print(len(generatorValuesEvery5mins))
    print(len(JanLoadEvery5mins))
    pickle.dump(generatorValuesEvery5mins, open("generatorValuesEvery5mins.pkl", "wb"))
    pickle.dump(JanLoadEvery5mins, open("JanLoadEvery5mins.pkl", "wb"))

