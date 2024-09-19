import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import os
import random
from sympy import *
from pandas import Series, DataFrame

mf_grc = np.load('mf_grc.npy', allow_pickle=True).item()
mf_goc = np.load('mf_goc.npy', allow_pickle=True).item()
goc_grc = np.load('goc_grc.npy', allow_pickle=True)
grc_seq = np.load('grc_seq.npy', allow_pickle=True)
grc_label = np.load('grc_label.npy', allow_pickle=True)

def main():
    test_input()
    Conditions = ['Control','ACh','Sero']
    for Condition in Conditions:
        print('Running {} condition'.format(Condition))
        trial_sim(Condition)
    print('done')


def test_input():
    def burst_spikes(r, min_spikes=1, max_spikes=5):
        alpha = 2 + r * 3
        beta = 5 - r * 3
        beta_value = np.random.beta(alpha, beta)
        scaled_spike_count = min_spikes + beta_value * (max_spikes - min_spikes)
        return int(round(scaled_spike_count))
    
    def generate_oscillatory_burst_times(T,r,amp):
        dt = 0.001
        min_interval = 0.05
        time = np.arange(0, T, dt)
        r_t = amp*r
        burst_times = []
        last_burst_time = -np.inf
        
        for t in time:
            rate = r_t[int(t / dt)]
            if t - last_burst_time >= min_interval:
                if np.random.rand() < rate * dt:
                    burst_times.append(t)
                    last_burst_time = t

        return burst_times

    def spike_input(T,amp,Hz,s,b,c):
        dt = 0.001
        time = np.arange(0, T, dt)
        r = (np.sin(Hz*np.pi * (time+s))-b)*c
        r = clip(r,0,1)
        burst_times = generate_oscillatory_burst_times(T, r, amp)
        spike_train = []
        for t in burst_times:
            r_b = (np.sin(Hz*np.pi * (t+s))-b)*c
            r_b = clip(r_b,0,1)
            spike_num = burst_spikes(r_b, min_spikes=1, max_spikes=5)
            for x in range(spike_num):
                spike_train.append(t+x*0.01)
            
        return np.round(spike_train, decimals=3)

    indices_1 = []
    spike_train_1 = []
    for x in range(500):
        spike_train = spike_input(10,40,2,0,0.6,2.5)
        spike_train_1.extend(spike_train)
        indices_1.extend(np.ones(len(spike_train))*x)
    spike_train_1 = spike_train_1*second

    indices_2 = []
    spike_train_2 = []
    for x in range(500):
        spike_train = spike_input(10,40,2,0.8,0.6,2.5)
        spike_train_2.extend(spike_train)
        indices_2.extend(np.ones(len(spike_train))*x)
    spike_train_2 = spike_train_2*second

    np.save('test_id1_10',indices_1)
    np.save('test_id2_10',indices_2)
    np.save('test_train1_10',spike_train_1)
    np.save('test_train2_10',spike_train_2)    

def trial_sim(Condition):
    @check_units(mu=1,sigma=1,result=1)
    def gamma_parms(mu,sigma):
        a,theta = symbols('a theta')
        eqns = [Eq(a*theta,mu), Eq(a*theta**2,sigma**2)]
        vars = [a,theta]
        sols = nonlinsolve(eqns,vars)
        sol_a,sol_theta = next(iter(sols))
        shape = float(sol_a)
        scale = float(sol_theta)
        return shape, scale
    
    runtime = 10000

    # Leak conductance
    g_l_GrC   = 0.2   * nS
    g_t_GrC   = 1.0   * nS
    g_l_GoC   = 3.    * nS

    # Reversal potential (leak, excitatory, inhibitory)
    E_l_GrC   = -75   * mV
    E_e_GrC   =   0   * mV
    E_i_GrC   = -75   * mV
    
    E_l_GoC   = -51   * mV
    E_e_GoC   =   0   * mV
    E_i_GoC   = -75   * mV

    # Membrane capacitance
    C_m_GrC   =  3.1  * pF
    C_m_GoC   = 60.   * pF

    # Decay constants
    tau_e_decay_GrC = 12.0 * ms
    tau_e_decay_GoC = 12.0 * ms
    
    tau_i_decay_GrC = 20.0 * ms
    tau_i_decay_GoC = 20.0 * ms
    
    # Absolute refractory period
    tau_r_GrC = 2 * ms
    tau_r_GoC = 10* ms
    
    # Spiking threshold
    V_th_GrC   = -55 * mV
    V_th_GoC   = -50 * mV
    
    # Resting potential
    V_r_GrC    = -75 * mV
    V_r_GoC    = -55 * mV
    
    # Golgi cell reset potential
    V_reset_GoC = -55 * mV
    
    # Synaptic weights
    w_i_GrC = 2.0 * nS
    w_i_GoC = 0.6 * nS
    fixed_w_e_GrC = 0.3 * nS
    w_e_GoC_GrC = 0.0 * nS
    fixed_w_e_GoC_M = 0.02 * nS
    
    # Stochastic fluctuating excitatory current
    sigma_n_GoC = 0.1 * nS
    sigma_n_GrC = 0.05 * nS
    tau_n   = 20 * ms
    
    #GrC equation
    GrC_eqs = '''
    dv/dt   = (g_l_GrC*(E_l_GrC-v) + (g_e+g_n)*(E_e_GrC-v) + (g_i+g_t)*(E_i_GrC-v))/C_m_GrC : volt (unless refractory)
    dg_e/dt = -g_e/tau_e_decay_GrC : siemens
    dg_i/dt = -g_i/tau_i_decay_GrC : siemens
    dg_n/dt = (-g_n + sigma_n_GrC * sqrt(tau_n) * xi)/tau_n : siemens
    g_e_tot_GrC1 : siemens
    g_e_tot_GrC2 : siemens
    g_i_tot_GrC : siemens
    g_t = reduce_tonic * g_t_GrC : siemens
    reduce_tonic: 1
    '''
    
    #GoC equation
    GoC_eqs = '''
    dv/dt   = (g_l_GoC*(E_l_GoC-v) + (g_e+g_n)*(E_e_GoC-v)+g_i*(E_i_GoC-v))/C_m_GoC : volt (unless refractory)
    dg_e/dt = -g_e/tau_e_decay_GoC : siemens
    dg_i/dt = -g_i/tau_i_decay_GoC : siemens
    dg_n/dt = (-g_n + sigma_n_GoC * sqrt(tau_n) * xi)/tau_n : siemens
    E_l_GoC : volt (constant)
    '''
    
    GrC = NeuronGroup(3000,
                      Equations(GrC_eqs,
                                        g_l   = g_l_GrC,
                                        E_l   = E_l_GrC,
                                        E_e   = E_e_GrC,
                                        E_i   = E_i_GrC,
                                        C_m   = C_m_GrC,
                                        tau_e = tau_e_decay_GrC,
                                        tau_i = tau_i_decay_GrC),
                      threshold  = 'v > V_th_GrC',
                      reset      = 'v = V_r_GrC',
                      refractory = 'tau_r_GrC',
                      method     = 'euler')
    GrC.v   = V_r_GrC
    GrC.reduce_tonic[:] = 1
    
    GoC = NeuronGroup(5,
                      Equations(GoC_eqs,
                                        g_l = g_l_GoC,
                                        E_l = E_l_GoC,
                                        E_e = E_e_GoC,
                                        E_i = E_i_GoC,
                                        C_m = C_m_GoC,
                                        tau_e = tau_e_decay_GoC),
                      threshold  = 'v > V_th_GoC',
                      reset      = 'v = V_r_GoC',
                      refractory = 'tau_r_GoC',
                      method     = 'euler')
    GoC.v   = V_r_GoC
    GoC.E_l_GoC = E_l_GoC
    
    #recall testing input spike train
    indices_1 = np.load('test_id1_10.npy', allow_pickle=True)
    indices_2 = np.load('test_id2_10.npy', allow_pickle=True)
    spike_train_1 = np.load('test_train1_10.npy', allow_pickle=True)
    spike_train_2 = np.load('test_train2_10.npy', allow_pickle=True)
    spike_train_1 = spike_train_1*second
    spike_train_2 = spike_train_2*second
    
    mf_input1 = SpikeGeneratorGroup(500, indices_1, spike_train_1)
    mf_input2 = SpikeGeneratorGroup(500, indices_2, spike_train_2)
    
    M1_GrC = Synapses(mf_input1,GrC,
                     model  = '''w_e_GrC : siemens
                                 g_e_tot_GrC1_post = g_e : siemens (summed)''',
                     on_pre = 'g_e += w_e_GrC')
    M1_GrC.connect(i = mf_grc['mf1'], j = mf_grc['grc1'])
    
    M2_GrC = Synapses(mf_input2,GrC,
                     model  = '''w_e_GrC : siemens
                                g_e_tot_GrC2_post = g_e : siemens (summed)''',
                     on_pre = 'g_e += w_e_GrC')
    M2_GrC.connect(i = mf_grc['mf2'], j = mf_grc['grc2'])
    
    GrC_GoC = Synapses(GrC,GoC,
                       model  = '''w_e_GoC : siemens''',
                       on_pre = 'g_e += w_e_GoC')
    for x in range(5):
        G_post = list(itertools.chain(*goc_grc[x]))
        GrC_GoC.connect(i = G_post, j = np.ones(len(G_post),dtype=int)*x)
        
    GoC_GrC = Synapses(GoC,GrC,
                       model  = '''w_i_GrC : siemens
                                   g_i_tot_GrC_post = g_i : siemens (summed)''',
                       on_pre = 'g_i += w_i_GrC',
                       delay = tau_r_GrC)
    for x in range(5):
        G_post = list(itertools.chain(*goc_grc[x]))
        GoC_GrC.connect(i = np.ones(len(G_post),dtype=int)*x, j = G_post)
    GoC_GrC.w_i_GrC = w_i_GrC
    
    GoC_GoC = Synapses(GoC,GoC,
                       model  = '''w_i_GoC : siemens''',
                       on_pre = 'g_i += w_i_GoC*(rand()<0.8)',
                       delay  = tau_r_GoC)
    GoC_GoC.connect(i = [0,0,1,1,2,2,3,3,4,4], j = [0,1,0,2,1,3,2,4,3,4])
    GoC_GoC.w_i_GoC = w_i_GoC
    
    M1_GoC = Synapses(mf_input1,GoC,
                      model   = '''w_e_GoC_M : siemens''',
                      on_pre  = 'g_e += w_e_GoC_M')
    M1_GoC.connect(i = mf_goc['mf1'], j = mf_goc['goc1'])
    M1_GoC.w_e_GoC_M[:] = fixed_w_e_GoC_M
    
    M2_GoC = Synapses(mf_input2,GoC,
                      model   = '''w_e_GoC_M : siemens''',
                      on_pre  = 'g_e += w_e_GoC_M')
    M2_GoC.connect(i = mf_goc['mf2'], j = mf_goc['goc2'])
    M2_GoC.w_e_GoC_M[:] = fixed_w_e_GoC_M
    
    @network_operation(dt=runtime*ms, when = 'start')
    def neuromodulation(t):
        global Condition
        if not str(Condition) == 'Control':
            if str(Condition) == 'ACh':
                GoC.E_l_GoC = -55 * mV
                M1_GrC.w_e_GrC *= 0.45
                M2_GrC.w_e_GrC *= 0.45
                M1_GoC.w_e_GoC_M *= 0.46
                M2_GoC.w_e_GoC_M *= 0.46
                reduction_shape,reduction_scale = gamma_parms(0.4,0.07)
                GrC.reduce_tonic[:] = np.random.RandomState(seed=2).gamma(reduction_shape,reduction_scale,3000)
                weight_changed = True
            
            if str(Condition) == 'Sero':
                GoC.E_l_GoC = -43 * mV
                GoC_GrC.w_i_GrC *= 1.2
                reduction_shape,reduction_scale = gamma_parms(1.6,0.18)
                GrC.reduce_tonic[:] = np.random.RandomState(seed=2).gamma(reduction_shape,reduction_scale,3000)
                weight_changed = True
        
            if str(Condition) == 'DA':
                M1_GrC.w_e_GrC *= 0.6
                M2_GrC.w_e_GrC *= 0.6
                M1_GoC.w_e_GoC_M *= 0.4
                M2_GoC.w_e_GoC_M *= 0.4            
                weight_changed = True
    
    #weight implementation
    weight1 = np.load('STDP_synaptic_weight1.npy', allow_pickle=True)
    weight2 = np.load('STDP_synaptic_weight2.npy', allow_pickle=True)
    weight1 = weight1[:,-1]
    weight2 = weight2[:,-1]
    weight3 = np.load('test_i.npy', allow_pickle=True)
    for pre, post, w in zip(mf_grc['mf1'], mf_grc['grc1'], weight1):
        M1_GrC.w_e_GrC[pre, post] = w * siemens
    for pre, post, w in zip(mf_grc['mf2'], mf_grc['grc2'], weight2):
        M2_GrC.w_e_GrC[pre, post] = w * siemens
    GrC_GoC.w_e_GoC = weight3*siemens
    
    #simulate the test
    M = SpikeMonitor(GrC)
    MO = SpikeMonitor(GoC)
    state_GrC  = StateMonitor(GrC, ['g_e_tot_GrC1','g_i_tot_GrC'], dt=1*ms, record = True)
    state_syn1  = StateMonitor(M1_GrC, 'w_e_GrC', dt=1*ms, record = True)
    state_syn2  = StateMonitor(M2_GrC, 'w_e_GrC', dt=1*ms, record = True)
    run(runtime*ms)

    spike_trains = M.spike_trains()
    max_length = max(len(spikes) for spikes in spike_trains.values())

    spike_train_array = np.full((len(spike_trains), max_length), np.nan)
    
    for i, (key, spikes) in enumerate(spike_trains.items()):
        spike_train_array[i, :len(spikes)] = spikes
    
    np.save('randomized_{}SpikeTrain'.format(Condition),spike_train_array)
    
