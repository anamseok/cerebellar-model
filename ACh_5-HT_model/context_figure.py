import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import os
import random
from sympy import *
from pandas import Series, DataFrame

mf_grc = np.load('mf_grc.npy', allow_pickle=True).item()
grc_seq = np.load('grc_seq.npy', allow_pickle=True)
control_all = np.load('randomized_ControlSpikeTrain.npy', allow_pickle=True)
ACh_all    = np.load('randomized_AChSpikeTrain.npy', allow_pickle=True)
Sero_all     = np.load('randomized_SeroSpikeTrain.npy', allow_pickle=True)
DA_all     = np.load('randomized_DASpikeTrain.npy', allow_pickle=True)

total_grc_seq = list(itertools.chain(*grc_seq.tolist()))      
def spike_sorting(spike_train_array):
    group_spike_data = spike_train_array[total_grc_seq]
    group_spike_data_nm = spike_train_array[[item for item in total_grc_seq if item in mf_grc['grc3']]]
    group_spike_data_norm = spike_train_array[[item for item in total_grc_seq if not item in mf_grc['grc3']]]
    return group_spike_data,group_spike_data_nm,group_spike_data_norm

grc_type_nm   = {}
grc_type_norm = {}
for group, num in enumerate(grc_seq.tolist()):
    for item in num:
        if item in mf_grc['grc3']:
            if group not in grc_type_nm:
                grc_type_nm[group] = []
            grc_type_nm[group].append(item)
        else:
            if group not in grc_type_norm:
                grc_type_norm[group] = []
            grc_type_norm[group].append(item)
control_spike_data,control_spike_data_nm,control_spike_data_norm = spike_sorting(control_all)
ach_spike_data,ach_spike_data_nm,ach_spike_data_norm = spike_sorting(ACh_all)
sero_spike_data,sero_spike_data_nm,sero_spike_data_norm = spike_sorting(Sero_all)
da_spike_data,da_spike_data_nm,da_spike_data_norm = spike_sorting(DA_all)

# Plotting function
def raster(data, tick_positions = None):
    fig, ax = plt.subplots(figsize=(20, 30))
    num_neurons = data.shape[0]

    for i in range(num_neurons):
        times = data[i, :]  # Get all time points for neuron i
        if not np.isnan(times[0]):
            valid_times = times[~np.isnan(times)]
            for time in valid_times:
                if 0.1 <= time % 1 < 0.3:
                    color = 'red'
                elif 0.3 <= time % 1 <= 0.4:
                    color = 'purple'
                elif 0.4 < time % 1 < 0.65:
                    color = 'blue'
                else:
                    color = 'black'  # Default color for times not in specified ranges

                ax.scatter(time, i, marker='|', color=color, s=10, linewidths=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron')
    ax.set_title('Raster Plot of Neuronal Firing Times')
    ax.set_xlim([0, 2])
    #ax.set_ylim(tick_positions[0],tick_positions[2])
    if tick_positions is not None:
        ax.set_yticks(tick_positions)
    ax.invert_yaxis()


    plt.show()
def combined_population(data_conditions1,data_conditions2):  
    conditions = ['Control', 'ACh', 'Sero', 'DA']
    colors = ['black', 'red', 'blue', 'purple']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=200)
    
    for data, condition, color in zip(data_conditions1, conditions, colors):
        rates, time_bins = calculate_firing_rates(data)
        ax1.plot(time_bins, rates, color=color, label=condition)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title('with NM signal')
    ax1.legend(loc='upper right')

    # Second plot (right side) for data_conditions2
    for data, condition, color in zip(data_conditions2, conditions, colors):
        rates, time_bins = calculate_firing_rates(data)
        ax2.plot(time_bins, rates, color=color, label=condition)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('without NM signal')
    ax2.legend(loc='upper right')
    
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    plt.tight_layout()
    plt.show()

def calculate_firing_rates(spikes):
    bin_width = 0.05 
    time_step = 0.01   
    total_time = 10
    seconds = np.arange(0, total_time, 1)
    all_average_rates = []

    for sec in seconds:
        time_bins = np.arange(sec+0.0000001, sec + 1 - bin_width + time_step, time_step)
        firing_rates = np.zeros((len(spikes), len(time_bins)))
        for x in range(len(spikes)):
            spike_times = np.nan_to_num(spikes[x])
            for i, t in enumerate(time_bins):
                count = np.sum((spike_times >= t) & (spike_times < t + bin_width))
                firing_rates[x, i] = count / (bin_width / second)
        average_rates = np.mean(firing_rates, axis=0)
        all_average_rates.append(average_rates)

    all_average_rates = np.array(all_average_rates)
    overall_average_rate = np.mean(all_average_rates, axis=0)
    time_bins = np.arange(0.0000001, 1 - bin_width + time_step, time_step)

    return overall_average_rate, time_bins

for x in range(13):
    data1 = [control_all[grc_type_nm[x]],ACh_all[grc_type_nm[x]],Sero_all[grc_type_nm[x]],DA_all[grc_type_nm[x]]]
    data2 = [control_all[grc_type_norm[x]],ACh_all[grc_type_norm[x]],Sero_all[grc_type_norm[x]],DA_all[grc_type_norm[x]]]
    combined_population(data1, data2)

