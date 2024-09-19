#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms import approximation
import os
import random


# In[2]:


#calculate bipartite modularity Q
def bipartite_modularity(G):
    adj_matrix = nx.bipartite.matrix.biadjacency_matrix(G, row_order=['mf{}'.format(i) for i in range(1000)], 
                                                        column_order=['GrC{}'.format(i) for i in range(3000)])
    a = adj_matrix.toarray()
    k = np.sum(a, axis = 1)
    d = np.sum(a, axis = 0)
    p = np.zeros((1000,3000))
    for i in range(len(k)):
        for j in range(len(d)):
            if i <500 and j<1500:
                p[i][j] = a[i][j]-k[i]*d[j]/12000 #total synapse connection : 12000
            elif i>499 and j>1499:
                p[i][j] = a[i][j]-k[i]*d[j]/12000
            else:
                p[i][j] = 0
                
    return sum(sum(p))/12000+0.5


# In[3]:


#basic MF-GrC connectivity based on bipartite network
def mix_net(s):
    MF_GL = nx.Graph()
    
    mf_labels = ['mf{}'.format(i) for i in range(0, 1030)]
    GrC_labels = ['GrC{}'.format(i) for i in range(0, 3000)]
    pre1 = []
    post1 = []
    pre2 = []
    post2 = []
    pre3 = []
    post3 = []    
    
    MF_GL.add_nodes_from(mf_labels, bipartite=0)  
    MF_GL.add_nodes_from(GrC_labels, bipartite=1)
    
    mix = np.random.choice(range(0, 3000), size=s, replace=False) #pick the GrCs which would be randomly wired(scale=s)
    target = np.random.choice(range(0, 3000), size=1500, replace=False) #pick the GrCs which would recieve NM related input(50%)
    
    for node in range(0, 3000):
        #randomly connect for picked GrCs
        if node in mix:
            connecting_nodes = np.random.choice(range(0, 1000), size=4, replace=False)
            for target_node in connecting_nodes:
                MF_GL.add_edge('mf{}'.format(target_node), 'GrC{}'.format(node))
                if target_node<500:
                    pre1.append(target_node)
                    post1.append(node)
                else:
                    pre2.append(target_node-500)
                    post2.append(node)
            
        #biased connect for unpicked GrCs
        else:
            if node<1500:
                connecting_nodes_1 = np.random.choice(range(0, 500), size=4, replace=False)
                for target_node in connecting_nodes_1:
                    MF_GL.add_edge('mf{}'.format(target_node), 'GrC{}'.format(node))
                    pre1.append(target_node)
                    post1.append(node)
            else:
                connecting_nodes_2 = np.random.choice(range(500, 1000), size=4, replace=False)
                for target_node in connecting_nodes_2:
                    MF_GL.add_edge('mf{}'.format(target_node), 'GrC{}'.format(node))
                    pre2.append(target_node-500)
                    post2.append(node)
                    
    #connect for NM related MF and GrCs     
    for x in range(30):
        node = x+1000
        target_nodes = target[x*50:(x+1)*50]
        for i in range(50):
            MF_GL.add_edge('mf{}'.format(node), 'GrC{}'.format(target_nodes[i]))
            pre3.append(node-1000)
            post3.append(target_nodes[i])

    mf_grc = {'mf1':pre1, 'mf2':pre2, 'mf3':pre3, 'grc1':post1, 'grc2':post2, 'grc3':post3}
    return MF_GL, mf_grc


# In[5]:


#categorize the GrCs based on the connectivity
def categorize_grc_nodes(G, grc_nodes, m1_nodes, m2_nodes):
    categories = {}
    for node in grc_nodes:
        m1_count = sum(1 for neighbor in G.neighbors(node) if neighbor in m1_nodes)
        m2_count = sum(1 for neighbor in G.neighbors(node) if neighbor in m2_nodes)
        if m1_count == 4:
            category = 0  # 4 from m1
        elif m1_count == 3 and m2_count == 1:
            category = 1  # 3 from m1, 1 from m2
        elif m1_count == 2 and m2_count == 2:
            category = 2  # 2 from m1, 2 from m2
        elif m1_count == 1 and m2_count == 3:
            category = 3  # 1 from m1, 3 from m2
        elif m2_count == 4:
            category = 4  # 4 from m2
        categories[node] = category

    nodes_grouped = {}
    for grc, node_number in categories.items():
        grc_number = int(grc[3:])
        if node_number not in nodes_grouped:
            nodes_grouped[node_number] = []
        nodes_grouped[node_number].append(grc_number)
    
    return categories, nodes_grouped


# In[7]:


#get MF-GoC connectivity
def mf_goc_connect():
    num_con = [12,9,6,3,0]
    pre_1   = []
    post_1  = []
    pre_2   = []
    post_2  = []
    #each GoC gets total 12 MF inputs
    for goc,mf in enumerate(num_con):
        if mf>0:
            pre_1.extend(np.random.choice(500,mf,replace = False))
            post_1.extend(np.ones(mf,dtype=int)*goc)
    for goc,mf in enumerate(num_con[::-1]):
        if mf>0:
            pre_2.extend(np.random.choice(500,mf,replace = False))
            post_2.extend(np.ones(mf,dtype=int)*goc)
    mf_goc = {'mf1':pre_1, 'mf2':pre_2, 'goc1':post_1, 'goc2':post_2}
    return mf_goc


# In[77]:


#get GoC-GrC connectivity
def goc_grc_connect(nodes_grouped):
    goc_grc = [[] for _ in range(5)]
    grc_seq = []
    grc_label = [0]
    num = 0

    for x in range(5):
        if x == 0 or x == 4:
            # Pick 80% for groups 0 and 4
            pick_ratio = 0.8
            pick = np.random.choice(nodes_grouped[x], int(len(nodes_grouped[x]) * pick_ratio), replace=False).tolist()
            non_picked = list(set(nodes_grouped[x]) - set(pick))

            if x == 0:
                # Append picked first for group 0
                append_order = [pick, non_picked]
            else:
                # Append non-picked first for group 4
                append_order = [non_picked, pick]
            
            for items in append_order:
                goc_grc[x if items is pick else abs(x-1)].append(items)
                grc_seq.append(items)
                num += len(items)
                grc_label.append(num)

        else:
            # Pick 60% for groups 1, 2, 3
            pick_ratio = 0.6
            pick = np.random.choice(nodes_grouped[x], int(len(nodes_grouped[x]) * pick_ratio), replace=False).tolist()
            non_picked = list(set(nodes_grouped[x]) - set(pick))
            non_picked_1 = np.random.choice(non_picked, int(len(non_picked) * 0.5), replace=False).tolist()
            non_picked_2 = list(set(non_picked) - set(non_picked_1))

            for group, items in zip([x-1, x, x+1], [non_picked_1, pick, non_picked_2]):
                goc_grc[group].append(items)
                grc_seq.append(items)
                num += len(items)
                grc_label.append(num)

    return goc_grc, grc_seq, grc_label


# In[4]:


MF_GL, mf_grc = mix_net(2000) #here scale for mix_net:2000, change as you want


# In[8]:


mf_goc = mf_goc_connect()


# In[78]:


goc_grc, grc_seq, grc_label = goc_grc_connect(nodes_grouped)


# In[6]:


#visualize the network
G = MF_GL
mf_nodes = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
grc_nodes = set(G) - mf_nodes
m1_nodes = {n for n in mf_nodes if int(n[2:]) < 500}
m2_nodes = {n for n in mf_nodes if int(n[2:]) >= 500 and int(n[2:]) <1000}
m3_nodes = {n for n in mf_nodes if int(n[2:]) >= 1000}

categories, nodes_grouped = categorize_grc_nodes(G, grc_nodes, m1_nodes, m2_nodes)

#categorized GrC nodes
pos = {node: (1, categories[node]*500 + index/10) for index, node in enumerate(grc_nodes)}

#MF nodes
pos.update((node, (0, index*1.5)) for index, node in enumerate(m1_nodes)) #put major sensorimotor MF at the left side
pos.update((node, (2, 1120+index*2)) for index, node in enumerate(m3_nodes)) #put NM related MF at the right side
pos.update((node, (0, 1550+index*1.5)) for index, node in enumerate(m2_nodes)) #put major sensorimotor MF at the left side


color_map = ['pink', '#DA70D6', 'purple', '#6495ED', 'skyblue']
# Drawing the graph
plt.figure(figsize=(60, 80))
nx.draw_networkx_nodes(G, pos, nodelist=m1_nodes, node_color='pink', label='m1_nodes', node_size=20)
nx.draw_networkx_nodes(G, pos, nodelist=m2_nodes, node_color='skyblue', label='m2_nodes', node_size=20)
nx.draw_networkx_nodes(G, pos, nodelist=m3_nodes, node_color='orange', label='m3_nodes', node_size=20)
nx.draw_networkx_nodes(G, pos, nodelist=grc_nodes, node_color=[color_map[categories[node]] for node in grc_nodes], label='GrC Nodes', node_size=20)

# Drawing edges
m1_edges = [(u, v) for u, v in G.edges() if u in m1_nodes]
m2_edges = [(u, v) for u, v in G.edges() if u in m2_nodes]
m3_edges = [(u, v) for u, v in G.edges() if u in m3_nodes]
nx.draw_networkx_edges(G, pos, edgelist=m1_edges, edge_color='pink', alpha=0.1)
nx.draw_networkx_edges(G, pos, edgelist=m2_edges, edge_color='skyblue', alpha=0.1)
nx.draw_networkx_edges(G, pos, edgelist=m3_edges, edge_color='orange', alpha=0.2)
plt.title('Bipartite Graph Representation')
plt.axis('off')
plt.show()


# In[82]:


np.save('mf_grc', np.array(mf_grc, dtype=object), allow_pickle=True)
np.save('mf_goc', np.array(mf_goc, dtype=object), allow_pickle=True)
np.save('goc_grc.npy', np.array(goc_grc, dtype=object), allow_pickle=True)
np.save('grc_seq.npy', np.array(grc_seq, dtype=object), allow_pickle=True)
np.save('grc_label.npy', grc_label)

