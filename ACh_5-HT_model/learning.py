#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms import approximation
from brian2 import *
import os
import random
from sympy import *
import pandas as pd
import scipy.sparse
import seaborn as sns
import itertools


# In[2]:


mf_grc = np.load('mf_grc.npy', allow_pickle=True).item()
mf_goc = np.load('mf_goc.npy', allow_pickle=True).item()
goc_grc = np.load('goc_grc.npy', allow_pickle=True)
grc_seq = np.load('grc_seq.npy', allow_pickle=True)
grc_label = np.load('grc_label.npy', allow_pickle=True)


# In[3]:


Condition = 'Control'


# In[4]:


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


# In[5]:


runtime = 10000

# Synaptic weight parameters
random_weights = bool(os.getenv("CHAT_RANDOM_WEIGHTS", True))
# weights_gamma_shape,weights_gamma_scale = gamma_parms(0.15,0.2)
weights_gamma_shape,weights_gamma_scale = gamma_parms(0.3,0.3)

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
V_reset_GoC = -55 * mV #-60

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

# symmetric STDP rule
M1 = 1.3
M2 = 1.1
M1_scale = 1.2
M2_scale = 1
sigma1 = 23
sigma2 = 26
sigma1_scale = 23
sigma2_scale = 26
S1 = M1/(sigma1*np.sqrt(2*np.pi))
S2 = M2/(sigma2*np.sqrt(2*np.pi))
S1_scale = M1_scale/(sigma1_scale*np.sqrt(2*np.pi))
S2_scale = M2_scale/(sigma2_scale*np.sqrt(2*np.pi))

bound = 3* nS
tau_w_n = 2000 * ms


# In[6]:


#GrC equation
GrC_eqs = '''
dv/dt   = (g_l_GrC*(E_l_GrC-v) + (g_e+g_n)*(E_e_GrC-v) + (g_i+g_t)*(E_i_GrC-v))/C_m_GrC : volt (unless refractory)
dg_e/dt = -g_e/tau_e_decay_GrC : siemens
dg_i/dt = -g_i/tau_i_decay_GrC : siemens
dg_n/dt = (-g_n + sigma_n_GrC * sqrt(tau_n) * xi)/tau_n : siemens
dw_scale/dt = -(w_scale - w_mean) / tau_w_n : 1
g_e_tot_GrC1 : siemens
g_e_tot_GrC2 : siemens
g_i_tot_GrC : siemens
g_t = reduce_tonic * g_t_GrC : siemens
reduce_tonic: 1
w_mean:1
'''

#GoC equation
GoC_eqs = '''
dv/dt   = (g_l_GoC*(E_l_GoC-v) + (g_e+g_n)*(E_e_GoC-v)+g_i*(E_i_GoC-v))/C_m_GoC : volt (unless refractory)
dg_e/dt = -g_e/tau_e_decay_GoC : siemens
dg_i/dt = -g_i/tau_i_decay_GoC : siemens
dg_n/dt = (-g_n + sigma_n_GoC * sqrt(tau_n) * xi)/tau_n : siemens
E_l_GoC : volt (constant)
'''


# In[7]:


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
GrC.w_mean = 1
GrC.w_mean[mf_grc['grc3']] = 0.8
GrC.w_scale = 1
GrC.w_scale[mf_grc['grc3']] = 0.8


# In[8]:


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


# In[9]:


def burst_spikes(r, min_spikes=1, max_spikes=5):
    # Scale alpha and beta to shift the distribution's mean based on r
    # Assume r is already normalized between 0 and 1
    alpha = 2 + r * 3  # Linearly increase alpha from 2 to 5 as r goes from 0 to 1
    beta = 5 - r * 3   # Linearly decrease beta from 5 to 2 as r goes from 0 to 1

    # Generate a value from the beta distribution
    beta_value = np.random.beta(alpha, beta)

    # Scale and shift the beta output to get spike count
    scaled_spike_count = min_spikes + beta_value * (max_spikes - min_spikes)

    # Return the integer number of spikes, rounding to nearest integer
    return int(round(scaled_spike_count))
def generate_oscillatory_burst_times(T,r,amp):
    """
    Generate burst times following a non-homogeneous Poisson process with minimum interval.

    :param T: Total time in seconds.
    :param dt: Time step in seconds.
    :param min_interval: Minimum interval between bursts in seconds.
    :return: List of burst times in seconds.
    """
    dt = 0.001
    min_interval = 0.05
    time = np.arange(0, T, dt)
    # Define the firing rate as sin^2(2*pi*time)
    r_t = amp*r
    
    burst_times = []
    last_burst_time = -np.inf  # Initialize far in the past
    
    # Simulate the process
    for t in time:
        rate = r_t[int(t / dt)]
        # Check if current time is eligible for next burst (after min_interval)
        if t - last_burst_time >= min_interval:
            # Poisson probability for a burst
            if np.random.rand() < rate * dt:
                burst_times.append(t)
                last_burst_time = t  # Update last burst time

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

#sensorimotor input1
indices_1 = []
spike_train_1 = []
for x in range(500):
    spike_train = spike_input(10,40,2,0,0.6,2.5)
    spike_train_1.extend(spike_train)
    indices_1.extend(np.ones(len(spike_train))*x)
spike_train_1 = spike_train_1*second

#sensorimotor input2
indices_2 = []
spike_train_2 = []
for x in range(500):
    spike_train = spike_input(10,40,2,0.8,0.6,2.5)
    spike_train_2.extend(spike_train)
    indices_2.extend(np.ones(len(spike_train))*x)
spike_train_2 = spike_train_2*second

#NM related input
indices_3 = []
spike_train_3 = []
for x in range(30):
    spike_train = spike_input(10,25,2,0.45,0.4,5/3)
    spike_train_3.extend(spike_train)
    indices_3.extend(np.ones(len(spike_train))*x)
spike_train_3 = spike_train_3*second


# In[10]:


mf_input1 = SpikeGeneratorGroup(500, indices_1, spike_train_1)
mf_input2 = SpikeGeneratorGroup(500, indices_2, spike_train_2)
mf_input3 = SpikeGeneratorGroup(30, indices_3, spike_train_3)


# In[11]:


M1_GrC = Synapses(mf_input1,GrC,
                 model='''w_e_GrC : siemens
                          last_prespike : second
                          last_postspike : second
                          g_e_tot_GrC1_post = g_e : siemens (summed)''',
                on_pre='''g_e += w_e_GrC
                          last_prespike = t
                          delta = (t-last_postspike)/ms
                          w_e_GrC += w_scale_post*(S1 * exp(-(delta**2) / (2 * sigma1**2)) - S2 * exp(-(delta**2) / (2 * sigma2**2)))*nS
                          w_e_GrC = clip(w_e_GrC, 0.005*nS, bound)''',
                on_post='''last_postspike = t
                           delta = (t-last_prespike)/ms
                           w_e_GrC += w_scale_post*(S1 * exp(-(delta**2) / (2 * sigma1**2)) - S2 * exp(-(delta**2) / (2 * sigma2**2)))*nS
                           w_e_GrC = clip(w_e_GrC, 0.005*nS, bound)
                           ''')
M1_GrC.connect(i = mf_grc['mf1'], j = mf_grc['grc1'])
M2_GrC = Synapses(mf_input2,GrC,
                 model='''w_e_GrC : siemens
                          last_prespike : second
                          last_postspike : second
                          g_e_tot_GrC2_post = g_e : siemens (summed)''',
                on_pre='''g_e += w_e_GrC
                          last_prespike = t
                          delta = (t-last_postspike)/ms
                          w_e_GrC += w_scale_post*(S1 * exp(-(delta**2) / (2 * sigma1**2)) - S2 * exp(-(delta**2) / (2 * sigma2**2)))*nS
                          w_e_GrC = clip(w_e_GrC, 0.005*nS, bound)''',
                on_post='''last_postspike = t
                           delta = (t-last_prespike)/ms
                           w_e_GrC += w_scale_post*(S1 * exp(-(delta**2) / (2 * sigma1**2)) - S2 * exp(-(delta**2) / (2 * sigma2**2)))*nS
                           w_e_GrC = clip(w_e_GrC, 0.005*nS, bound)
                           ''')
M2_GrC.connect(i = mf_grc['mf2'], j = mf_grc['grc2'])
M3_GrC = Synapses(mf_input3,GrC,
                 model='''w_e_GrC : siemens
                          last_prespike : second
                          last_postspike : second
                          ''',
                on_pre='''g_e += w_e_GrC
                          last_prespike = t
                          delta = (t-last_postspike)/ms
                          w_scale_post += (S1_scale * exp(-(delta**2) / (2 * sigma1_scale**2)) - S2_scale * exp(-(delta**2) / (2 * sigma2_scale**2)))*80
                          w_scale_post = clip(w_scale_post, 0, 2)''',
                on_post='''last_postspike = t
                           delta = (t-last_prespike)/ms
                           w_scale_post += (S1_scale * exp(-(delta**2) / (2 * sigma1_scale**2)) - S2_scale * exp(-(delta**2) / (2 * sigma2_scale**2)))*80
                           w_scale_post = clip(w_scale_post, 0, 2)
                           ''')
M3_GrC.connect(i = mf_grc['mf3'], j = mf_grc['grc3'])
M3_GrC.w_e_GrC[:] = fixed_w_e_GrC #without learning, NM related MF-GrC synapse weights are fixed

GrC_GoC = Synapses(GrC,GoC,
                   model  = '''w_e_GoC : siemens''',
                   on_pre = 'g_e += w_e_GoC')
for x in range(5):
    G_post = list(itertools.chain(*goc_grc[x]))
    GrC_GoC.connect(i = G_post, j = np.ones(len(G_post),dtype=int)*x)
weight3 = np.load('test_i.npy', allow_pickle=True)
GrC_GoC.w_e_GoC = weight3*siemens

GoC_GrC = Synapses(GoC,GrC,
                   model  = '''w_i_GrC : siemens
                               g_i_tot_GrC_post = g_i : siemens (summed)''',
                   on_pre = 'g_i += w_i_GrC',
                   delay  = tau_r_GrC)
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


# In[12]:


#randomize weight
@network_operation(dt=runtime*ms, when='start')
def update_input():
    weight_sigma = 0.2
    tempidx = 0
    for i in range(500):
        weight = np.random.gamma(2,0.1,len(M1_GrC.w_e_GrC[i,:]))*nS
        M1_GrC.w_e_GrC[i,:] = weight
        tempidx += 1
    for i in range(500):
        weight = np.random.gamma(2,0.1,len(M2_GrC.w_e_GrC[i,:]))*nS
        M2_GrC.w_e_GrC[  i,:] = weight
        tempidx += 1
    tempidx = 0
    weight_means = np.random.RandomState(seed=int(os.getenv("CHAT_SEED_WEIGHT_MEANS", 20))).gamma(weights_gamma_shape,weights_gamma_scale,3000)
    for i in range(3000):
        mu = weight_means[tempidx]
        sigma = weight_sigma
        shape,scale = gamma_parms(mu,sigma)
        weight = np.random.RandomState(seed=trial_seed).gamma(shape,scale)*nS/16
        GrC_GoC.w_e_GoC[i,:] = weight
        tempidx += 1


# In[14]:


M = SpikeMonitor(GrC)
MO = SpikeMonitor(GoC)
state_GrC  = StateMonitor(GrC, ['g_e_tot_GrC1','g_i_tot_GrC','w_scale'], dt=1*ms, record = True)
state_syn1  = StateMonitor(M1_GrC, 'w_e_GrC', dt=1*ms, record = True)
state_syn2  = StateMonitor(M2_GrC, 'w_e_GrC', dt=1*ms, record = True)
trial_seed = 45

run(runtime*ms)

spike_trains = M.spike_trains()
max_length = max(len(spikes) for spikes in spike_trains.values())

#get GrC spike train
spike_train_array = np.full((len(spike_trains), max_length), np.nan)

for i, (key, spikes) in enumerate(spike_trains.items()):
    spike_train_array[i, :len(spikes)] = spikes

#save synaptic weights for neuromodulator change    
np.save('STDP_synaptic_weight1',state_syn1.w_e_GrC)
np.save('STDP_synaptic_weight2',state_syn2.w_e_GrC)
np.save('test_i',GrC_GoC.w_e_GoC)


# In[15]:


#GoC firing plot
plt.figure(figsize=(10, 2))
plt.plot(MO.t/ ms, MO.i,'|', markersize=10)
plt.xlim(0,10000)
plt.ylim(min(MO.i)-0.5, max(MO.i)+0.5)
plt.show


# In[16]:


#input(MF) firing plot
spike_monitor1 = SpikeMonitor(mf_input1)
spike_monitor2 = SpikeMonitor(mf_input2)
spike_monitor3 = SpikeMonitor(mf_input3)

run(runtime*ms)

spike_times1 = spike_monitor1.t / ms
spike_indices1 = spike_monitor1.i

spike_times2 = spike_monitor2.t / ms
spike_indices2 = spike_monitor2.i

spike_times3 = spike_monitor3.t / ms
spike_indices3 = spike_monitor3.i

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axs[0].plot(spike_times1, spike_indices1,'.', color='lightcoral', markersize=1)
axs[0].set_title('Input Module 1')
axs[0].set_ylabel('Neuron index')

axs[1].plot(spike_times2, spike_indices2,'.' ,color='lightblue', markersize=1)
axs[1].set_title('Input Module 2')
axs[1].set_xlabel('Time (ms)')

axs[2].plot(spike_times3, spike_indices3,'.' ,color='black', markersize=1)
axs[2].set_title('NM related Input')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Neuron index')

plt.xlim(0,2000)
plt.tight_layout()
plt.show()

