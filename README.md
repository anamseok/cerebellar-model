# cerbellar neural network
This repository is about cerebellar system model : Golgi-Mossy Fiber-Granule cell layer <br/>
## Acetylcholine and Serotonin treated cerebellar system model
This model designed to invest spiking activity of granule cells in normal/Acetylcholine/Serotonin Condition. <br/> 
Network of Normal and Acetylcholine Condition is based on "https://github.com/trfore/chatmodel.git"<br/>
For Serotonin(5-HT), further variables are referd in: <br/>
>1. Fleming, E., & Hull, C. (2019). Serotonin regulates dynamics of cerebellar granule cell activity by modulating tonic inhibition. Journal of neurophysiology, 121(1), 105-114.<br/>
>2. Regehr, W. G. (2012). Identification of an inhibitory circuit that regulates cerebellar Golgi cell activity. Neuron, 73(1), 149-158.
### Fixed parameters in Control
| Cell Type | V<sub>r</sub>(mV) | V<sub>th</sub>(mV) | C<sub>m</sub>(pF) | G<sub>l</sub>(nS) | G<sub>t</sub>(nS) | E<sub>l</sub>(mV) | E<sub>e</sub>(mV) | E<sub>i</sub>(mV) | σ<sub>n</sub>(nS) | Ƭ<sub>n</sub>(ms) | Ƭ<sub>e</sub>(ms) | Ƭ<sub>i</sub>(ms) |
| --------- | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------------ | ---------- | ---------- | ---------- |
| GrC       |   -75   |   -55    |   3.1   |   0.2   |    1    |   -75   |    0    |   -75   |     0.05     |     20     |     12     |     20     |
| GoC       |   -55   |   -50    |   60    |    3    |    0    |   -51   |    0    |   -75   |      0.1     |     20     |     12     |     0      |
### Changed parameters in Acetylcholine and Serotonin
| Condition | GoC - E<sub>l</sub>(mV) |     GrC - tonic inhibition    | Ƭ<sub>i</sub>(ms) |
| --------- | ----------------------- | ----------------------------- | ----------------- |
|    ACh    |          -55            | N(0.4,0.07) _(40% reduction)_ |     no change     |
|    5-HT   |          -47            | N(1.6, 0.18) _(60% enhanced)_ |         20        |
* As no researches are founded mentioning GABAa receptor decay time constant of Golgi cell, rescaling decay time constant of Golgi cell in 5-HT condition is refered from well known Lugaro cell- Golgi synapse, GABAergic slow IPSC.
### Inhibitory Golgi-Golgi synapse in Serotonin condition
In 5-HT treated condition, negative feedback of Golgi cell to other Golgi cells limiting excessive inhibition of Golgi cell to Granule cell layer are implemented through inhibitory Golgi-Golgi synapses.<br/>
5-HT specific Golgi-Golgi synapse properties are described below:
* 20% are connected among Golgi cells
* average synaptic weight : 0.33±0.08 nS
* IPSC failure rate : ~20%
### References
* (https://doi.org/10.1016/j.neuron.2011.10.030)
* (https://doi.org/10.1152/jn.00492.2018)
* (https://doi.org/10.1523/JNEUROSCI.2148-19.2020)
* (https://doi.org/10.1523/JNEUROSCI.21-16-06045.2001)
