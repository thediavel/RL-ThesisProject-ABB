import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
import enlopy as el
import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandapower.control as ct
pd.options.display.float_format = '{:.4g}'.format

#### fix lineIndex in ieee-4
class powerGrid_ieee4:
    def __init__(self):
        print('in init. Here we lay down the grid structure and load some random state values based on IEEE dataset');
        with open('JanLoadEvery5mins.pkl', 'rb') as pickle_file:
            self.loadProfile = pickle.load(pickle_file)
        with open('generatorValuesEvery5mins.pkl', 'rb') as pickle_file:
            self.powerProfile = pickle.load(pickle_file)

        self.k_old=0;
        self.q_old=0;
        self.actionSpace = {'v_ref_pu': [i*5 / 100 for i in range(16, 25)], 'lp_ref': [i * 5 for i in range(0, 31)]}
        ## Basic ieee 4bus system
        self.net = pp.networks.case4gs();
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

        trafo_control = ct.DiscreteTapControl(net=self.net, tid=0, vm_lower_pu=0.95, vm_upper_pu=1.05)

        # Breaker between grid HV bus and trafo HV bus to connect buses
        sw_SVC = pp.create_switch(self.net, bus=1, element=0, et='t', type='CB', closed=False)
        # Shunt devices connected with MV bus
        shuntDev = pp.create_shunt(self.net, bus_SVC, 0, in_service=True, name='Shunt Device', step=1)
        # TSC1 = pp.create_shunt(self.net, bus_SVC, -50, in_service=True, name='TSC1', step=1)
        # TSC2 = pp.create_shunt(self.net, bus_SVC, -50, in_service=True, name='TSC2', step=1)
        # TCR = pp.create_shunt(self.net, bus_SVC, 1, in_service=True, name='TCR', step=80)
        ####Series device (at line 3, in middle between bus 2 and 3)
        # Add intermediate buses for bypass and series compensation impedance
        bus_SC1 = pp.create_bus(self.net, name='SC bus 1', vn_kv=230, type='n', geodata=(-1, 3.1), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        bus_SC2 = pp.create_bus(self.net, name='SC bus 2', vn_kv=230, type='n', geodata=(-1, 3.0), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        sw_SC_bypass = pp.create_switch(self.net, bus=5, element=6, et='b', type='CB', closed=True)
        imp_SC = pp.create_impedance(self.net, from_bus=5, to_bus=6, rft_pu=0.01272, xft_pu=-0.0636,
                                     rtf_pu=0.01272, xtf_pu=-0.0636, sn_mva=250, in_service=True)
        # Adjust orginal Line 3 to connect to new buses instead.
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
        self.net.gen.drop(index=[0], inplace=True)  # Drop PV generator
        pp.create_sgen(self.net, 3, p_mw=318, q_mvar=181.4, name='static generator', scaling=1)
        self.stateIndex = np.random.randint(len(self.loadProfile)-1, size=1)[0];
        #self.stateIndex=0
        self.scaleLoadAndPowerValue(self.stateIndex,-1);
        try:
            pp.runpp(self.net, run_control=False)
            print('Environment has been successfully initialized');
        except:
            print('Some error occured while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def takeAction(self, lp_ref, v_ref_pu):
        #q_old = 0
        bus_index_shunt = 1
        line_index=3;
        impedenceBackup = self.net.impedance.loc[0, 'xtf_pu'];
        shuntBackup = self.net.shunt.q_mvar
        self.net.switch.at[1, 'closed'] = False
        self.net.switch.at[0, 'closed'] = True
        ##shunt compenstation
        q_comp = self.Shunt_q_comp(v_ref_pu, bus_index_shunt, self.q_old);
        self.q_old = q_comp;
        self.net.shunt.q_mvar =  q_comp;
        ##series compensation
        #i_ref = self.net.res_line.loading_percent[line_index];
        k_x_comp_pu = self.K_x_comp_pu(lp_ref, 1, self.k_old);
        self.k_old = k_x_comp_pu;
        x_line_pu = self.X_pu(line_index)
        self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = x_line_pu * k_x_comp_pu
        networkFailure = False

        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1);
            try:
                pp.runpp(self.net, run_control=True);
                reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
            except:
                print('Unstable environment settings');
                networkFailure = True;
                reward = -1000;

        return (self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]), reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """
        try:
            pp.runpp(self.net);
            reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
        except:
            networkFailure=True;
            self.net.shunt.q_mvar=shuntBackup;
            self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']]=impedenceBackup;
            pp.runpp(self.net);
            reward=1000;
            return self.net.res_bus,reward,True;
        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            if (self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1) == False):
                networkFailure = True;
                reward = 1000;
                # self.stateIndex -= 1;
        return self.net.res_bus, reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """
    ##Function to calculate line reactance in pu
    def X_pu(self, line_index):
        s_base = 100e6
        v_base = 230e3
        x_base = pow(v_base, 2) / s_base
        x_line_ohm = self.net.line.x_ohm_per_km[line_index]
        x_line_pu = x_line_ohm / x_base  # Can take one since this line is divivded into
        # 2 identical lines with length 0.5 km
        return x_line_pu

    def takeAction_old(self,p_ref_pu,v_ref_pu):
        # print('taking action p_ref_pu='+str(p_ref_pu)+' and observe the grid for reactance component or power losses at stateIndex:');
        # print(self.stateIndex)
        S_base = 100e6
        V_base = 230e3
        x_base = math.pow(V_base, 2) / S_base
        x_line_pu =self.net.line.x_ohm_per_km[3] / x_base
        k_x_comp_pu = self.K_x_comp_pu(p_ref_pu,x_line_pu)
        k_x_comp_pu=-0.8 if k_x_comp_pu<-0.8 else k_x_comp_pu;
        k_x_comp_pu = 0.2 if k_x_comp_pu > 0.2 else k_x_comp_pu;
        if self.net.switch.at[1, 'closed']:
            self.net.switch.at[1, 'closed'] = False
            self.net.switch.at[0, 'closed'] = True
        impedenceBackup=self.net.impedance.loc[0, 'xtf_pu'];
        shuntBackup=self.net.shunt.q_mvar
        self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = x_line_pu * k_x_comp_pu
        q_comp = self.Shunt_q_comp( v_ref_pu)
        self.net.shunt.q_mvar = q_comp
        networkFailure=False
        try:
            pp.runpp(self.net);
            reward = self.calculateReward(self.net.res_bus.vm_pu);
        except:
            networkFailure=True;
            self.net.shunt.q_mvar=shuntBackup;
            self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']]=impedenceBackup;
            pp.runpp(self.net);
            reward=1000;
            return self.net.res_bus,reward,True;

        self.stateIndex+=1;
        if self.stateIndex < len(self.powerProfile):
            if(self.scaleLoadAndPowerValue(self.stateIndex,self.stateIndex-1) == False):
                networkFailure=True;
                reward=1000;
                #self.stateIndex -= 1;
        return self.net.res_bus,reward,self.stateIndex == len(self.powerProfile) or networkFailure;

    def reset(self):
        print('reset the current environment for next episode');
        oldIndex = self.stateIndex;
        self.stateIndex = np.random.randint(len(self.loadProfile)-1, size=1)[0];
        self.net.switch.at[0, 'closed'] = False
        self.net.switch.at[1, 'closed'] = True
        self.k_old = 0;
        self.q_old = 0;
        self.scaleLoadAndPowerValue(self.stateIndex,oldIndex);
        try:
            pp.runpp(self.net, run_control=False);
            print('Environment has been successfully initialized');
        except:
            print('Some error occurred while resetting the environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def calculateReward(self, voltages, loadingPercent):
        rew = 0;
        for i in range(0, len(voltages)):
            if voltages[i] > 1.25 or voltages[i] < 0.8:
                rew -= 50;
            elif voltages[i] > 1.05 or voltages[i] < 0.95:
                rew -= 15;
            else:
                rew += 20;
        rew = rew / len(voltages)
        loadingPercentInstability = np.std(loadingPercent) * len(loadingPercent);
        # print(loadingPercent)
        # print(loadingPercentInstability)
        return rew - loadingPercentInstability;

    def plotGridFlow(self):
        print('plotting powerflow for the current state')
        plot.simple_plot(self.net)

    def scaleLoadAndPowerValue(self,index,refIndex):
        if refIndex != -1:
            scalingFactorLoad = self.loadProfile[index]/self.loadProfile[refIndex];
            scalingFactorPower = self.powerProfile[index] / self.powerProfile[refIndex];
        else:
            scalingFactorLoad = self.loadProfile[index] / (sum(self.loadProfile)/len(self.loadProfile));
            scalingFactorPower = self.powerProfile[index] / max(self.powerProfile);

        self.net.load.p_mw[1] = self.net.load.p_mw[1] * scalingFactorLoad;
        self.net.load.q_mvar[1] = self.net.load.q_mvar[1] * scalingFactorLoad;
        self.net.sgen.p_mw = self.net.sgen.p_mw * scalingFactorPower;
        self.net.sgen.q_mvar = self.net.sgen.q_mvar * scalingFactorPower;

    ##Function for transition from reference power to reactance of "TCSC"
    def K_x_comp_pu(self, loading_perc_ref, line_index, k_old):
        ##NEW VERSION TEST:
        c = 15  # Coefficient for transition tuned to hit equal load sharing at nominal IEEE
        k_x_comp_max_ind = 0.4
        k_x_comp_max_cap = -k_x_comp_max_ind
        loading_perc_meas = self.net.res_line.loading_percent[line_index]
        k_delta = (c * k_x_comp_max_ind * (
                    loading_perc_meas - loading_perc_ref) / 100) - k_old  # 100 To get percentage in pu
        k_x_comp = k_delta + k_old

        # Bypassing series device if impedance close to 0
        if abs(k_x_comp) < 0.0001:  # Helping with convergence
            self.net.switch.closed[1] = True  # ACTUAL network, not a copy

        if k_x_comp > k_x_comp_max_ind:
            k_x_comp = k_x_comp_max_ind
        if k_x_comp < k_x_comp_max_cap:
            k_x_comp = k_x_comp_max_cap
        return k_x_comp

    def Shunt_q_comp(self, v_ref_pu, bus_index, q_old):
        v_bus_pu = self.net.res_bus.vm_pu[bus_index]
        k = 25  # Coefficient for transition, tuned to hit 1 pu with nominal IEEE
        q_rated = 100  # Mvar
        q_min = -q_rated
        q_max = q_rated
        q_delta = k * q_rated * (
                    v_bus_pu - v_ref_pu) - q_old  # q_old might come in handy later with RL if able to take actions without
        # independent change in environment
        q_comp = q_delta + q_old

        if q_comp > q_max:
            q_comp = q_max
        if q_comp < q_min:
            q_comp = q_min

        # print(q_comp)
        return q_comp

    def K_x_comp_pu_old(self, p_ref_pu,x_line_pu):
        v_s_pu = self.net.res_bus.vm_pu[3]
        v_r_pu = self.net.res_bus.vm_pu[2]
        # 2 identical lines with length 0.5 km
        delta_deg = self.net.res_bus.va_degree[3] - self.net.res_bus.va_degree[2]
        k_x_comp = ((v_s_pu * v_r_pu * math.sin(math.radians(delta_deg))) / (p_ref_pu * x_line_pu)) - 1
        return k_x_comp

    ## Function for transition from reference parameter to reactive power output of shunt device
    def Shunt_q_comp_old(self, v_ref_pu):
        v_bus_pu = self.net.res_bus.vm_pu[1]
        k = 25 #Coefficient for transition, tuned to hit 1 pu with nominal IEEE
        q_rated = 100 #Mvar
        q_min = -q_rated
        q_max = q_rated
        q_comp = k*q_rated*(v_bus_pu-v_ref_pu);
        if q_comp > q_max:
            q_comp = q_max
        if q_comp < q_min:
            q_comp = q_min
        return q_comp


class powerGrid_ieee2:
    def __init__(self):
        #print('in init. Here we lay down the grid structure and load some random state values based on IEEE dataset');
        with open('JanLoadEvery5mins.pkl', 'rb') as pickle_file:
            self.loadProfile = pickle.load(pickle_file)
        with open('generatorValuesEvery5mins.pkl', 'rb') as pickle_file:
            self.powerProfile = pickle.load(pickle_file)

        self.actionSpace = {'v_ref_pu': [i*5 / 100 for i in range(16, 25)], 'lp_ref': [i * 15 for i in range(0, 11)]}
        self.k_old = 0;
        self.q_old = 0;
        ## Basic ieee 4bus system
        net_temp=pp.networks.case4gs();
        # COPY PARAMETERS FROM TEMP NETWORK TO USE IN 2 BUS RADIAL SYSTEM.
        # BUSES
        b0_in_service = net_temp.bus.in_service[0]
        b0_max_vm_pu = net_temp.bus.max_vm_pu[0]
        b0_min_vm_pu = net_temp.bus.min_vm_pu[0]
        b0_name = net_temp.bus.name[0]
        b0_type = net_temp.bus.type[0]
        b0_vn_kv = net_temp.bus.vn_kv[0]
        b0_zone = net_temp.bus.zone[0]
        b0_geodata = (3, 2)

        b1_in_service = net_temp.bus.in_service[1]
        b1_max_vm_pu = net_temp.bus.max_vm_pu[1]
        b1_min_vm_pu = net_temp.bus.min_vm_pu[1]
        b1_name = net_temp.bus.name[1]
        b1_type = net_temp.bus.type[1]
        b1_vn_kv = net_temp.bus.vn_kv[1]
        b1_zone = net_temp.bus.zone[1]
        b1_geodata = (4, 2)

        # BUS ELEMENTS
        load_bus = net_temp.load.bus[1]
        load_in_service = net_temp.load.in_service[1]
        load_p_mw = net_temp.load.p_mw[1]
        load_q_mvar = net_temp.load.q_mvar[1]
        load_scaling = net_temp.load.scaling[1]

        extGrid_bus = net_temp.ext_grid.bus[0]
        extGrid_in_service = net_temp.ext_grid.in_service[0]
        extGrid_va_degree = net_temp.ext_grid.va_degree[0]
        extGrid_vm_pu = net_temp.ext_grid.vm_pu[0]
        extGrid_max_p_mw = net_temp.ext_grid.max_p_mw[0]
        extGrid_min_p_mw = net_temp.ext_grid.min_p_mw[0]
        extGrid_max_q_mvar = net_temp.ext_grid.max_q_mvar[0]
        extGrid_min_q_mvar = net_temp.ext_grid.min_q_mvar[0]

        # LINES
        line0_scaling = 1
        line0_c_nf_per_km = net_temp.line.c_nf_per_km[0]
        line0_df = net_temp.line.df[0]
        line0_from_bus = net_temp.line.from_bus[0]
        line0_g_us_per_km = net_temp.line.g_us_per_km[0]
        line0_in_service = net_temp.line.in_service[0]
        line0_length_km = net_temp.line.length_km[0]
        line0_max_i_ka = net_temp.line.max_i_ka[0]
        line0_max_loading_percent = net_temp.line.max_loading_percent[0]
        line0_parallel = net_temp.line.parallel[0]
        line0_r_ohm_per_km = net_temp.line.r_ohm_per_km[0] * line0_scaling
        line0_to_bus = net_temp.line.to_bus[0]
        line0_type = net_temp.line.type[0]
        line0_x_ohm_per_km = net_temp.line.x_ohm_per_km[0] * line0_scaling

        line1_scaling = 1.2
        line1_c_nf_per_km = line0_c_nf_per_km
        line1_df = line0_df
        line1_from_bus = line0_from_bus
        line1_g_us_per_km = line0_g_us_per_km
        line1_in_service = line0_in_service
        line1_length_km = line0_length_km
        line1_max_i_ka = line0_max_i_ka
        line1_max_loading_percent = line0_max_loading_percent
        line1_parallel = line0_parallel
        line1_r_ohm_per_km = line0_r_ohm_per_km
        line1_to_bus = line0_to_bus
        line1_type = line0_type
        line1_x_ohm_per_km = line0_x_ohm_per_km * line1_scaling

        ## creating 2 bus system using nominal values from ieee4 bus system
        self.net = pp.create_empty_network()
        # Create buses
        b0 = pp.create_bus(self.net, in_service=b0_in_service, max_vm_pu=b0_max_vm_pu, min_vm_pu=b0_min_vm_pu,
                           name=b0_name, type=b0_type, vn_kv=b0_vn_kv, zone=b0_zone, geodata=b0_geodata)

        b1 = pp.create_bus(self.net, in_service=b1_in_service, max_vm_pu=b1_max_vm_pu, min_vm_pu=b1_min_vm_pu,
                           name=b1_name, type=b1_type, vn_kv=b1_vn_kv, zone=b1_zone, geodata=b1_geodata)

        # Create bus elements
        load = pp.create_load(self.net, bus=load_bus, in_service=load_in_service,
                              p_mw=load_p_mw, q_mvar=load_q_mvar, scaling=load_scaling)

        extGrid = pp.create_ext_grid(self.net, bus=extGrid_bus, in_service=extGrid_in_service,
                                     va_degree=extGrid_va_degree,
                                     vm_pu=extGrid_vm_pu, max_p_mw=extGrid_max_p_mw, min_p_mw=extGrid_min_p_mw,
                                     max_q_mvar=extGrid_max_q_mvar, min_q_mvar=extGrid_min_q_mvar)

        # Create lines
        l0 = pp.create_line_from_parameters(self.net, c_nf_per_km=line0_c_nf_per_km, df=line0_df, from_bus=line0_from_bus,
                                            g_us_per_km=line0_g_us_per_km, in_service=line0_in_service,
                                            length_km=line0_length_km,
                                            max_i_ka=line0_max_i_ka, max_loading_percent=line0_max_loading_percent,
                                            parallel=line0_parallel, r_ohm_per_km=line0_r_ohm_per_km,
                                            to_bus=line0_to_bus,
                                            type=line0_type, x_ohm_per_km=line0_x_ohm_per_km)

        l1 = pp.create_line_from_parameters(self.net, c_nf_per_km=line1_c_nf_per_km, df=line1_df, from_bus=line1_from_bus,
                                            g_us_per_km=line1_g_us_per_km, in_service=line1_in_service,
                                            length_km=line1_length_km,
                                            max_i_ka=line1_max_i_ka, max_loading_percent=line1_max_loading_percent,
                                            parallel=line1_parallel, r_ohm_per_km=line1_r_ohm_per_km,
                                            to_bus=line1_to_bus,
                                            type=line1_type, x_ohm_per_km=line1_x_ohm_per_km)

        ####Shunt FACTS device (bus 1)
        # MV bus
        bus_SVC = pp.create_bus(self.net, name='MV SVCtrafo bus', vn_kv=69, type='n', geodata=(4.04, 1.98), zone=2,
                                max_vm_pu=1.1,
                                min_vm_pu=0.9)
        # Trafo
        trafoSVC = pp.create_transformer_from_parameters(self.net, hv_bus=1, lv_bus=2, in_service=True,
                                                         name='trafoSVC', sn_mva=110, vn_hv_kv=230, vn_lv_kv=69,
                                                         vk_percent=12, vkr_percent=0.26, pfe_kw=55, i0_percent=0.06,
                                                         shift_degree=0, tap_side='hv', tap_neutral=0, tap_min=-9,
                                                         tap_max=9,
                                                         tap_step_percent=1.5, tap_step_degree=0,
                                                         tap_phase_shifter=False)
        trafo_control = ct.DiscreteTapControl(net=self.net, tid=0, vm_lower_pu=0.95, vm_upper_pu=1.05)
        # Breaker between grid HV bus and trafo HV bus to connect buses
        sw_SVC = pp.create_switch(self.net, bus=1, element=0, et='t', type='CB', closed=False)
        # Shunt devices connected with MV bus
        shuntDev = pp.create_shunt(self.net, bus_SVC, 2, in_service=True, name='Shunt Device', step=1)

        ####Series device (at line 1, in middle between bus 0 and 1)
        # Add intermediate buses for bypass and series compensation impedance
        bus_SC1 = pp.create_bus(self.net, name='SC bus 1', vn_kv=230, type='n', geodata=(3.48, 2.05),
                                zone=2, max_vm_pu=1.1, min_vm_pu=0.9)
        bus_SC2 = pp.create_bus(self.net, name='SC bus 2', vn_kv=230, type='n', geodata=(3.52, 2.05),
                                zone=2, max_vm_pu=1.1, min_vm_pu=0.9)
        sw_SC_bypass = pp.create_switch(self.net, bus=3, element=4, et='b', type='CB', closed=True)
        imp_SC = pp.create_impedance(self.net, from_bus=3, to_bus=4, rft_pu=0.0000001272, xft_pu=-0.0636,
                                     rtf_pu=0.0000001272, xtf_pu=-0.0636, sn_mva=250,
                                     in_service=True)  # Just some default values
        # Adjust orginal Line 3 to connect to new buses instead.
        self.net.line.at[1, ['length_km', 'to_bus', 'name']] = [0.5, 3, 'line1_SC']
        lineSC2 = pp.create_line_from_parameters(self.net, name='line2_SC', c_nf_per_km=self.net.line.at[1, 'c_nf_per_km'],
                                                 df=self.net.line.at[1, 'df'], from_bus=4,
                                                 g_us_per_km=self.net.line.at[1, 'g_us_per_km'],
                                                 in_service=self.net.line.at[1, 'in_service'], length_km=0.5,
                                                 max_i_ka=self.net.line.at[1, 'max_i_ka'],
                                                 max_loading_percent=self.net.line.at[1, 'max_loading_percent'],
                                                 parallel=self.net.line.at[1, 'parallel'],
                                                 r_ohm_per_km=self.net.line.at[1, 'r_ohm_per_km'],
                                                 std_type=self.net.line.at[1, 'std_type'], to_bus=1,
                                                 type=self.net.line.at[1, 'type'],
                                                 x_ohm_per_km=self.net.line.at[1, 'x_ohm_per_km'])

        ## select a random state for the episode
        self.stateIndex = np.random.randint(len(self.loadProfile)-1, size=1)[0];
        #self.stateIndex=100
        self.scaleLoadAndPowerValue(self.stateIndex,-1);
        #pp.runpp(self.net, run_control=False);
        #print(self.net.res_bus.vm_pu);
        try:
            pp.runpp(self.net, run_control=False);
            print('Environment has been successfully initialized');
        except:
            print('Some error occurred while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def runEnv(self):
        try:
            pp.runpp(self.net, run_control=False);
            #print('Environment has been successfully initialized');
        except:
            print('Some error occurred while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def getCurrentState(self):
        bus_index_shunt = 1
        line_index = 1;
        return (self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]);

    def takeAction(self, lp_ref, v_ref_pu):
        #q_old = 0
        bus_index_shunt = 1
        line_index=1;
        impedenceBackup = self.net.impedance.loc[0, 'xtf_pu'];
        shuntBackup = self.net.shunt.q_mvar
        self.net.switch.at[1, 'closed'] = False
        self.net.switch.at[0, 'closed'] = True

        ##shunt compenstation
        q_comp = self.Shunt_q_comp(v_ref_pu, bus_index_shunt, self.q_old);
        self.q_old=q_comp;
        self.net.shunt.q_mvar =  q_comp;
        ##series compensation
        k_x_comp_pu = self.K_x_comp_pu(lp_ref, 1, self.k_old);
        self.k_old = k_x_comp_pu;
        x_line_pu=self.X_pu(line_index)
        self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = x_line_pu * k_x_comp_pu
        networkFailure = False

        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1);
            try:
                pp.runpp(self.net, run_control=True);
                reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
            except:
                print('Unstable environment settings');
                networkFailure = True;
                reward = -10000;
        else:
            print('wrong block!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return (self.net.res_bus.vm_pu[bus_index_shunt],self.net.res_line.loading_percent[line_index]), reward, self.stateIndex == len(self.powerProfile)-1 or networkFailure;

        """
        try:
            pp.runpp(self.net);
            reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
        except:
            networkFailure=True;
            self.net.shunt.q_mvar=shuntBackup;
            self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']]=impedenceBackup;
            pp.runpp(self.net);
            reward=1000;
            return self.net.res_bus,reward,True;
        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            if (self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1) == False):
                networkFailure = True;
                reward = 1000;
                # self.stateIndex -= 1;
        return self.net.res_bus, reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """
    ##Function to calculate line reactance in pu
    def X_pu(self, line_index):
        s_base = 100e6
        v_base = 230e3
        x_base = pow(v_base, 2) / s_base
        x_line_ohm = self.net.line.x_ohm_per_km[line_index]
        x_line_pu = x_line_ohm / x_base  # Can take one since this line is divivded into
        # 2 identical lines with length 0.5 km
        return x_line_pu

    def reset(self):
        #print('reset the current environment for next episode');
        oldIndex = self.stateIndex;
        self.stateIndex = np.random.randint(len(self.loadProfile) - 1, size=1)[0];
        #self.stateIndex=3729;
        self.net.switch.at[0, 'closed'] = False
        self.net.switch.at[1, 'closed'] = True
        self.k_old = 0;
        self.q_old = 0;
        self.scaleLoadAndPowerValue(self.stateIndex,oldIndex);
        try:
            pp.runpp(self.net, run_control=False);
            #print('Environment has been successfully initialized');
        except:
            print('Some error occurred while resetting the environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def calculateReward(self, voltages, loadingPercent):
        try:
            rew=0;
            for i in range(1,2):
                if voltages[i] > 1.25 or voltages[i] < 0.8:
                    rew -= 50;
                elif voltages[i] > 1.05 or voltages[i] < 0.95:
                    rew -= 15;
                else :
                    rew += 20;
            rew = rew;
            loadingPercentInstability=np.std(loadingPercent) * len(loadingPercent);
        except:
            print('exception in calculate reward')
            #print(voltages);
            #print(loadingPercent)
            return 0;
        return rew - loadingPercentInstability;

    def plotGridFlow(self):
        print('plotting powerflow for the current state')
        plot.simple_plot(self.net)

    def scaleLoadAndPowerValue(self,index,refIndex):
        if refIndex != -1:
            scalingFactorLoad = 0 if self.loadProfile[refIndex] == 0 else self.loadProfile[index]/self.loadProfile[refIndex];
            scalingFactorPower = 0 if self.powerProfile[refIndex]==0 else self.powerProfile[index] / self.powerProfile[refIndex];
        else:
            scalingFactorLoad = self.loadProfile[index] / (sum(self.loadProfile)/len(self.loadProfile));
            scalingFactorPower = self.powerProfile[index] / max(self.powerProfile);

        self.net.load.p_mw[0] = self.net.load.p_mw[0] * scalingFactorLoad;
        self.net.load.q_mvar[0] = self.net.load.q_mvar[0] * scalingFactorLoad;
        #self.net.sgen.p_mw = self.net.sgen.p_mw * scalingFactorPower;
        #self.net.sgen.q_mvar = self.net.sgen.q_mvar * scalingFactorPower;


    ##Function for transition from reference power to reactance of "TCSC"
    def K_x_comp_pu(self, loading_perc_ref, line_index, k_old):
        ##NEW VERSION TEST:
        c = 5  # Coefficient for transition tuned to hit equal load sharing at nominal IEEE
        #print((loading_perc_ref,line_index,k_old))
        k_x_comp_max_ind = 0.4
        k_x_comp_max_cap = -k_x_comp_max_ind
        loading_perc_meas = self.net.res_line.loading_percent[line_index]
        k_delta = (c * k_x_comp_max_ind * (
                    loading_perc_meas - loading_perc_ref) / 100) - k_old  # 100 To get percentage in pu
        k_x_comp = k_delta + k_old

        # Bypassing series device if impedance close to 0
        if abs(k_x_comp) < 0.0001:  # Helping with convergence
            self.net.switch.closed[1] = True  # ACTUAL network, not a copy

        if k_x_comp > k_x_comp_max_ind:
            k_x_comp = k_x_comp_max_ind
        if k_x_comp < k_x_comp_max_cap:
            k_x_comp = k_x_comp_max_cap
        return k_x_comp

    def Shunt_q_comp(self, v_ref_pu, bus_index, q_old):
        v_bus_pu = self.net.res_bus.vm_pu[bus_index]
        k = 10  # Coefficient for transition, tuned to hit 1 pu with nominal IEEE
        q_rated = 100  # Mvar
        q_min = -q_rated
        q_max = q_rated
        q_delta = k * q_rated * (
                    v_bus_pu - v_ref_pu) - q_old  # q_old might come in handy later with RL if able to take actions without
        # independent change in environment
        q_comp = q_delta + q_old

        if q_comp > q_max:
            q_comp = q_max
        if q_comp < q_min:
            q_comp = q_min

        # print(q_comp)
        return q_comp


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
        randomValue_prob = np.random.random();
        if randomValue > windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] or randomValue_prob < 0.4:
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

#createLoadProfile()