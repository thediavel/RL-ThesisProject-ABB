import pandapower.control as ct

# SHUNT CONTROLLER
class ShuntFACTS(ct.basic_controller.Controller):
    def __init__(self, net, busVoltageInd, convLim, shuntIndex=0, q_mvar_rating=50, max_iter=30, in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        # construct through superclass
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)

        # Initialise class variables
        self.shuntIndex = shuntIndex
        self.busVoltageInd = busVoltageInd
        self.ref = 1.0  # reference value to reach
        self.convLim = convLim  # limit threshold for convergence
        self.meas = self.net.res_bus.vm_pu[busVoltageInd]
        self.applied = False
        self.q_mvar_max = q_mvar_rating
        self.q_mvar_min = -q_mvar_rating
        self.iter_counter = 0  # count number of iterations
        self.maxed_counter = 0  # To count iterations if maxed out and cant converge to v_ref
        self.max_iter = max_iter # maximum umber of iterations
        self.v_delta = 0
        self.v_delta_accum = 0

    # return boolean for if controled has converged to ref value
    def is_converged(self):
        self.meas = self.net.res_bus.vm_pu[self.busVoltageInd]
        # Converged if within limit or output maxed for three iterations without convergence
        if abs(self.meas - self.ref) < self.convLim or self.maxed_counter >= 4 or self.iter_counter == self.max_iter:
            self.applied = True
        return self.applied

    # In case the controller is not yet converged, the control step is executed.
    def control_step(self):
        # Measurement
        self.meas = self.net.res_bus.vm_pu[self.busVoltageInd]
        self.v_delta = self.meas - self.ref

        # Control Coefficients
        K_p = 10  # Factor 10 is to cap at rating when v_delta is +/- 0.1 pu.
        K_i = 15

        # PI-control equation
        self.net.shunt.q_mvar[self.shuntIndex] = K_p * self.q_mvar_max * (
            self.v_delta) + K_i * self.q_mvar_max * self.v_delta_accum

        # Make sure output don't exceed rating
        if self.net.shunt.q_mvar[self.shuntIndex] + 0.00001 >= self.q_mvar_max:
            self.net.shunt.q_mvar[self.shuntIndex] = self.q_mvar_max
            self.maxed_counter += 1
        elif self.net.shunt.q_mvar[self.shuntIndex] - 0.00001 <= self.q_mvar_min:
            self.net.shunt.q_mvar[self.shuntIndex] = self.q_mvar_min
            self.maxed_counter += 1

        # Update for posible next iter of control
        #print('SHUNT:')
        #print(self.v_delta_accum)
        self.v_delta_accum += self.v_delta
        self.iter_counter += 1

        #print(self.net.shunt.q_mvar[self.shuntIndex])
        #print(self.meas)
        #print(self.iter_counter)
        #print(self.maxed_counter)

    # Finalize function MIGHT BE NEEDED IF RESET OF SOME CLASS VARIABLES NEEDED: DEPENDS ON HOW CALLED IN MAIN MODEL
    def finalize_control(self):
        self.applied = False
        self.v_delta_accum = 0
        self.iter_counter = 0  # count number of iterations
        self.maxed_counter = 0  # To count iterations if maxed out and cant converge to v_ref


# Series CONTROLLER
class SeriesFACTS(ct.basic_controller.Controller):
    def __init__(self, net, lineLPInd, convLim, x_line_pu, max_iter=30, switchInd=1, serIndex=0, x_comp_rating=0.4,
                 in_service=True,
                 recycle=False, order=0, level=0, **kwargs):
        # construct through superclass
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         initial_powerflow=True, **kwargs)

        # Initialise class variables
        self.switchInd = switchInd
        self.x_line_pu = x_line_pu
        self.serIndex = serIndex
        self.lineLPInd = lineLPInd
        self.ref = 50  # reference value to reach
        self.convLim = convLim  # limit threshold for convergence
        self.meas = 0
        self.applied = False
        self.x_comp_max = x_comp_rating
        self.x_comp_min = -x_comp_rating
        self.iter_counter = 0  # count number of iterations
        self.maxed_counter = 0  # To count iterations if maxed out and cant converge to v_ref
        self.max_iter = max_iter
        self.lp_delta = 0
        self.lp_delta_accum = 0

    # return boolean for if controled has converged to ref value
    def is_converged(self):
        self.meas = self.net.res_line.loading_percent[self.lineLPInd]
        # Converged if within limit or output maxed for three iterations without convergence
        if abs(self.meas - self.ref) < self.convLim or self.maxed_counter >= 4 or self.iter_counter == self.max_iter:
            self.applied = True
        return self.applied

    # In case the controller is not yet converged, the control step is executed.
    def control_step(self):
        # Measurement
        self.meas = self.net.res_line.loading_percent[self.lineLPInd]
        self.lp_delta = (self.meas - self.ref) / 100  # div by 100 to get value between 0-1

        # Control Coefficients
        K_p = 20
        K_i = 15

        # PI-control equation
        op = self.x_line_pu * (K_p * self.x_comp_max * (self.lp_delta) + K_i * self.x_comp_max * self.lp_delta_accum)

        # Make sure output don't exceed rating
        if op + 0.00001 >= self.x_comp_max:
            op = self.x_comp_max
            self.maxed_counter += 1
        elif op - 0.00001 <= self.x_comp_min:
            op = self.x_comp_min
            self.maxed_counter += 1

        # Set output of device
        self.net.impedance.loc[self.serIndex, ['xft_pu', 'xtf_pu']] = op

        # Bypassing series device if impedance close to 0
        if abs(op) < 0.0001:  # Helping with convergence
            self.net.switch.closed[self.switchInd] = True  # ACTUAL network, not a copy

        # Update for posible next iter of control
        #print('SERIES:')
        #print(self.lp_delta_accum)
        self.lp_delta_accum += self.lp_delta
        self.iter_counter += 1

        #print(op)
        #print(self.meas)
        #print(self.iter_counter)
        #print(self.maxed_counter)

    # Finalize function MIGHT BE NEEDED IF RESET OF SOME CLASS VARIABLES NEEDED: DEPENDS ON HOW CALLED IN MAIN MODEL
    def finalize_control(self):
        self.applied = False
        self.lp_delta_accum = 0
        self.iter_counter = 0  # count number of iterations
        self.maxed_counter = 0  # To count iterations if maxed out and cant converge to v_ref