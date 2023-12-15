import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KernelDensity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import rc

# fonttype setting required for CorelDraw compatibility
rc('pdf',fonttype=42)
def main():
    controlnorm, seronorm, achnorm, max_ach, max_sero, \
    control, ach, sero, \
    unique, counts, \
    int_g_e, int_g_i, \
    reduce_tonic = run_analysis()

    plot_spike_prob(controlnorm, seronorm, achnorm, max_ach, max_sero, control, ach, sero)

###########################################################
# Run analysis and return data required to create figures #
###########################################################
def run_analysis():
    # Load arrays listing all active MF->GrC pairs
    control_connections = np.load('controlConnections.npy', allow_pickle=True)
    ach_connections     = np.load('achConnections.npy', allow_pickle=True)
    # Check to make sure connections are the same for both conditions
    shared_connections = control_connections == ach_connections
    if np.all(shared_connections):
        connections = control_connections
    else:
        print('WARNING: Connections are not the same between conditions')

    # Calculate number of unique granule cells receiving active MF input
    GrC = connections[:,1]
    unique,counts = np.unique(GrC,return_counts=True)

    # Load spike probability arrays
    control_all = np.load('randomized_controlSpikeProb.npy', allow_pickle=True)
    sero_all    = np.load('randomized_seroSpikeProb.npy', allow_pickle=True)
    ach_all     = np.load('randomized_achSpikeProb.npy', allow_pickle=True)

    nt = control_all.shape[1]

    # Only consider unique GrCs receiving active MF input
    control_unique = control_all[unique,:]
    sero_unique    = sero_all[unique,:]
    ach_unique     = ach_all[unique,:]

    # Get rid of entirely quiescent cells in control
    indices_nonzero = ~np.all(control_unique==0,axis=1)
    control_nonzero = control_unique[indices_nonzero,1000:1060]
    sero_nonzero    = sero_unique[indices_nonzero,1000:1060]
    ach_nonzero     = ach_unique[indices_nonzero,1000:1060]

    # Update cell counts
    unique = unique[indices_nonzero]
    counts = counts[indices_nonzero]

    # Now select cells for spiking in control that matches experiment
    max_stim_two   = np.max(control_nonzero[:,18:22],axis=1)
    max_stim_three = np.max(control_nonzero[:,28:32],axis=1)

    selection_criteria = (max_stim_two <= 0.6) & (max_stim_three <= max_stim_two)

    control = control_nonzero[selection_criteria,:]
    sero    = sero_nonzero[selection_criteria,:]
    ach     = ach_nonzero[selection_criteria,:]

    # Update cell counts again
    unique = unique[selection_criteria]
    counts = counts[selection_criteria]

    # Normalize to control condition (max-min)
    controlmax = np.max(control,axis=1)
    controlnorm = control / controlmax[:,None]
    seronorm    = sero / controlmax[:,None]
    achnorm     = ach / controlmax[:,None]

    # Maximum normalized spike probability in muscarine
    max_ach = np.max(achnorm,axis=1)
    max_sero = np.max(seronorm,axis=1)


    # Load synaptic weight dictionaries
    control_weights     = np.load('controlWeights.npy', allow_pickle=True)
    ach_weights         = np.load('achWeights.npy', allow_pickle=True)
    x = control_weights.item()
    y = ach_weights.item()

    # Check to make sure weights are consistent for both conditions
    shared_weights = {k: x[k] for k in x if k in y and x[k]==y[k]}
    if len(shared_weights) < len(x):
        print('WARNING: Weights are not the same between conditions')
    else:
        weights = x

    # Load excitatory and inhibitory conductance traces
    g_e = np.load('controlCondE.npy', allow_pickle=True)
    g_i = np.load('controlCondI.npy', allow_pickle=True)
    if g_e.shape[1] != nt:
        print('WARNING: StateMonitor sampling does not match resolution of spiking data')
    else:
        start = 1000
        stop  = 1060
    # Make sure you only saved traces for unique GrCs receiving active MF input
    if g_e.shape[0] != control_unique.shape[0]:
        print('WARNING: Did not save excitatory conductances for only unique GrCs')
    else:
        g_e = g_e[indices_nonzero,start:stop]
        g_e = g_e[selection_criteria,:]

    if g_i.shape[0] != control_unique.shape[0]:
        print('WARNING: Did not save inhibitory conductances for only unique GrCs')
    else:
        g_i = g_i[indices_nonzero,start:stop]
        g_i = g_i[selection_criteria,:]

    #Integrate conductance traces
    nsamp = stop-start
    xval  = np.linspace(0,60,nsamp)
    int_g_e = np.trapz(g_e,xval,axis=1)
    int_g_i = np.trapz(g_i,xval,axis=1)

    # Load array with the fraction of tonic inhibition in muscarine
    reduce_tonic = np.load('tonicReduction.npy', allow_pickle=True)
    if len(reduce_tonic) != control_unique.shape[0]:
        print('WARNING: Did not save tonic reduction for only unique GrCs')
    else:
        reduce_tonic = reduce_tonic[indices_nonzero]
        reduce_tonic = reduce_tonic[selection_criteria]

    return controlnorm, seronorm, achnorm, max_ach, max_sero, \
           control, ach, sero, \
           unique, counts, \
           int_g_e, int_g_i, \
           reduce_tonic

########################################################
# Code to construct spike probability figure;          #
# Panel 1: Normalized spike probability plot for cells #
# with increasing probability in muscarine.            #
# Panel 2: Normalized spike probability plot for cells #
# with decreasing probaility in muscarine.             #
# Panel 3: Summary scatter plot of peak probability    #
# in muscarine vs. control                             #
########################################################
def plot_spike_prob(controlnorm, seronorm, achnorm, max_ach, max_sero, control, ach, sero):
    ach_increase = max_ach > 1
    ach_decrease  = max_ach < 1
    n_increase = np.sum(ach_increase)
    n_decrease = np.sum(ach_decrease)

    mean_control_up = np.mean(controlnorm[ach_increase,:],axis=0)
    mean_sero_up    = np.mean(seronorm[ach_increase,:],axis=0)
    mean_ach_up     = np.mean(achnorm[ach_increase,:],axis=0)

    mean_control_down = np.mean(controlnorm[ach_decrease,:],axis=0)
    mean_sero_down    = np.mean(seronorm[ach_decrease,:],axis=0)
    mean_ach_down     = np.mean(achnorm[ach_decrease,:],axis=0)

    nt = controlnorm.shape[1]

    stim_times = [1010, 1020, 1030]

    cond = max_ach != 1
    conds = max_sero != 1

    fig, ax = plt.subplots(1, 4, figsize=(14,5))
    fig.tight_layout()
    # Need this backend for figures to be compatible with CorelDraw
    with PdfPages('spike_prob_with_sero.pdf') as pdf:
        ax[0].plot(np.arange(nt)-10,mean_control_up,'k')
        ax[0].plot(np.arange(nt)-10,mean_sero_up,'r')
        ax[0].plot(np.arange(nt)-10,mean_ach_up,'b')
        bottom,top = ax[0].get_ylim()
        ax[0].text(0,top-0.5,'n={}'.format(n_increase))
        ax[0].legend(['Control','Serotonin','Muscarine'])
        ax[0].set_xlabel('time (ms)')
        ax[0].set_ylabel('spike probability (norm.)')

        ax[1].plot(np.arange(nt)-10,mean_control_down,'k')
        ax[1].plot(np.arange(nt)-10,mean_sero_down,'r')
        ax[1].plot(np.arange(nt)-10,mean_ach_down,'b')
        bottom,top = ax[1].get_ylim()
        ax[1].text(0,top-0.2,'n={}'.format(n_decrease))
        ax[1].set_xlabel('time (ms)')

        x = np.max(control[cond],axis=1)
        y = np.max(ach[cond], axis=1)
        c = Counter(zip(x,y))
        s = [2*c[(i,j)] for i,j in zip(x,y)]
        ax[2].scatter(x,y,c='k',s=s)
        ax[2].set_xlabel('Control peak spike probability')
        ax[2].set_ylabel('Muscarine peak spike probability')
        start,stop = ax[2].get_xlim()
        x = np.linspace(start,stop)
        y = x
        ax[2].plot(x,y,'--k')

        x = np.max(control[conds],axis=1)
        y = np.max(sero[conds], axis=1)
        c = Counter(zip(x,y))
        s = [2*c[(i,j)] for i,j in zip(x,y)]
        ax[3].scatter(x,y,c='k',s=s)
        ax[3].set_xlabel('Control peak spike probability')
        ax[3].set_ylabel('Serotonin peak spike probability')
        start,stop = ax[3].get_xlim()
        x = np.linspace(start,stop)
        y = x
        ax[3].plot(x,y,'--k')
        
        pdf.savefig()
        plt.close()
        
if __name__ == '__main__':
    main()
