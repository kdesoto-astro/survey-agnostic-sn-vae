import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
import torch.nn as nn

def masked_softmax(vec, mask, dim=1, epsilon=1e-10):
    if type(vec) == torch.Tensor: # had to hack this to avoid segfaults
        masked_vec = vec * mask.float()
        exps = torch.exp(masked_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    else:
        masked_vec = vec * mask
        exps = np.exp(masked_vec)
        masked_exps = exps * mask
        masked_sums = np.expand_dims(masked_exps.sum(axis=1), 1) + epsilon

    final_vec = masked_exps/masked_sums + ((1-mask) * vec)
    return final_vec

def custom_hier_loss(output, target, target_weights, mask_list, pathlengths):
	final_sum = 0
	alpha = 0.5
	output[:, 0] = 1.0

	for i,mask in enumerate(mask_list):
		output = masked_softmax(output, mask)
	output = output.log()
	output = output * np.exp(-alpha * (pathlengths - 1))
	final_sum = (target_weights * (output*target).sum(dim=1)).mean()
	#final_sum = ((output*target).sum(dim=1)).mean()

	return -final_sum

def calc_class_weights(labels, vertices, all_paths):
	# Get the weights of each class label...
	ulabels, class_counts = np.unique(labels, return_counts = True)
	class_weight_dict = {}
	for ulabel in ulabels:
		count_for_ulabel = 0
		for ulabel2 in ulabels:
			gind = np.where(vertices == ulabel2)
			if ulabel in all_paths[gind[0][0]]:
				count_for_ulabel += class_counts[np.where(ulabels==ulabel2)]

		class_weight_dict[ulabel] = (np.sum(class_counts)/(len(ulabels) * count_for_ulabel))
	return class_weight_dict

#this could take just a graph...
def calc_path_and_mask(G, vertices, root):
	all_paths = np.zeros((len(vertices), len(vertices)))
	new_new_A = np.zeros((len(vertices), len(vertices)))

	pathlengths = []
	parent_groups = []
	all_paths = []
	for i,node in enumerate(vertices):
		pathlengths.append(len(nx.shortest_path(G, root, node)))
		all_paths.append(nx.shortest_path(G, root, node))
		for thing in nx.shortest_path(G, root, node):
			gind = np.where(thing == np.asarray(vertices, dtype='str'))[0]
			new_new_A[i,gind[0]] = 1
		if i == 0:
			parent_groups.append(-1)
		else:
			parent_groups.append(np.where(np.asarray(vertices, dtype='str')==next(G.predecessors(node)))[0][0])
	#Make parent groups into a set of masks
	mask_list = []
	for pg in np.unique(parent_groups):
		if pg == -1:
			continue
		gind = np.where(parent_groups == pg)
		mask = np.zeros(len(new_new_A))
		mask[gind] = 1
		mask_list.append(torch.tensor(mask, dtype=int))
	y_dict = dict(zip(vertices, new_new_A))
	pathlengths = torch.tensor(pathlengths)

	return all_paths, pathlengths, mask_list, y_dict

def get_prob(input_vec, desired_class, all_paths, vertices, mask_list):
	output = input_vec * 1.0
	output[:,0] = 1.0

	for i,mask in enumerate(mask_list):
		output = masked_softmax(output, mask)
	
	gind = np.where(vertices == desired_class)
	myprob = torch.ones(len(input_vec))
	for thing in all_paths[gind[0][0]]:
		gind2 = np.where(vertices == thing)[0]
		myprob = myprob * output[:,gind2[0]]
	return myprob

def is_parent(label_child, desired_parent, vertices, all_paths):
	gind  = np.where(vertices == label_child)
	if desired_parent in all_paths[gind[0][0]]:
		return True
	else:
		return False

class Feedforward(torch.nn.Module):
		def __init__(self, input_size, hidden_size, output_size):
			super(Feedforward, self).__init__()
			self.input_size = input_size
			self.hidden_size  = hidden_size
			self.output_size  = output_size

			self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
			self.relu = torch.nn.Sigmoid()
			self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
			self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
		def forward(self, x):
			hidden = self.fc1(x)
			relu = self.relu(hidden)
			output1 = self.fc2(relu)
			output2 = self.relu(output1)
			output3 = self.fc3(output2)
			return output3


#I shamelessly stole this from stackoverflow: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

# Create graph of all kinds of transients so that we have whatever types we need.
G=nx.DiGraph()

G.add_edge('Object','StellarTransient')
G.add_edge('Object','SN-like')


G.add_edge('StellarTransient','AGN')
G.add_edge('AGN','QSO')

G.add_edge('StellarTransient', 'CV')
G.add_edge('StellarTransient', 'Nova')
G.add_edge('StellarTransient', 'Varstar')

G.add_edge('SN-like', 'SN')
G.add_edge('SN-like', 'ILOT')
G.add_edge('ILOT', 'LBV')
G.add_edge('ILOT', 'ILRT')
G.add_edge('ILOT', 'LRN')
G.add_edge('SN-like', 'TDE')


G.add_edge('SN', 'SN Ia')
G.add_edge('SN Ia', 'SN Ia-91T')
G.add_edge('SN Ia', 'SN Ia-91bg-like')
G.add_edge('SN Ia', 'SN Ia-CSM')
G.add_edge('SN Ia', 'SN Ia-SC')
G.add_edge('SN Ia', 'SN Ia-pec')
G.add_edge('SN Ia', 'SN Iax')


G.add_edge('SN', 'CC')

G.add_edge('CC', 'SLSN-I')
G.add_edge('CC', 'SN II')

G.add_edge('CC', 'SN Ibc')
G.add_edge('SN Ibc', 'SN Ib')
G.add_edge('SN Ibc', 'SN Ib-pec')
G.add_edge('SN Ibc', 'SN Ibn')
G.add_edge('SN Ibc', 'SN Ic')
G.add_edge('SN Ibc', 'SN Ic-BL')
G.add_edge('SN Ibc', 'SN Ic-Ca-rich')
G.add_edge('SN Ibc', 'SN Ic-pec')
G.add_edge('SN Ibc', 'SN Icn')

G.add_edge('CC', 'SLSN-II')
G.add_edge('SN II', 'SN II-pec')
G.add_edge('SN II', 'SN IIb')
G.add_edge('CC', 'SN IIn')

pos = hierarchy_pos(G, 'Object')

vertices = ['AGN', 'CV', 'ILRT', 'LBV', 'LRN', 'Nova', 'QSO', 'SLSN-I',
       'SLSN-II', 'SN', 'SN II', 'SN II-pec', 'SN IIb', 'SN IIn', 'SN Ia',
       'SN Ia-91T', 'SN Ia-91bg-like', 'SN Ia-CSM', 'SN Ia-SC',
       'SN Ia-pec', 'SN Iax', 'SN Ib', 'SN Ib-pec', 'SN Ibc', 'SN Ibn',
       'SN Ic', 'SN Ic-BL', 'SN Ic-Ca-rich', 'SN Ic-pec', 'SN Icn', 'TDE',
       'Varstar']
vertices = np.append(vertices, ['StellarTransient', 'SN-like', 'ILOT', 'CC'])
vertices = np.insert(vertices, 0, 'Object')

# uncomment to display graph of taxonomy
# fig = plt.figure(1, figsize=(20, 10))

# nx.draw_networkx(G, pos=pos, nodelist=vertices, node_color='white', with_labels=False, node_size=2000, arrows=False)
# text = nx.draw_networkx_labels(G, pos)
# for _, t in text.items():
#     t.set_rotation(45) 
#     plt.show()
