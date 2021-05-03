from matplotlib import pyplot as plt #plotting with the matrix plotting library
import networkx as nx #plotting graphical networks to visualize expert and inferred causal structures
from pgmpy.readwrite import BIFReader, BIFWriter #Save trained models as a Bayesian Inference File (.bif) and load them later 
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, State
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.estimators import HillClimbSearch, BicScore
from os import system


#Helper methods for network diagnostics
def PrintCPDs(model, message):
	cpdList = model.get_cpds()
	print(message+"\n")
	for n,cpd in enumerate(cpdList):
		nodeList = list(model.nodes())
		print("\n\n")
		print("CPD for node "+nodeList[n]+"\n")
		print(cpd)
		print("\n\n")
	
def SaveGraph(model, filename, fmt="PNG"): #saves the inferred and expert models as .png image files in working directory
	nodes = model.nodes()
	edges = model.edges()
	graph = nx.DiGraph() #instantiate a DiGraph (Directed Graph) object
	graph.add_nodes_from(nodes) #take node list from pgmpy and send it to DiGraph object
	graph.add_edges_from(edges) #take edges list from pgmpy and send it to DiGraph object
	nx.draw_networkx(graph, arrows=True) #draw the Bayesian Network
	plt.tight_layout() #fit all plotted objects within viewing frame
	plt.savefig(filename, format=fmt) #save in working directory as .png
	plt.clf() #close matplotlib plot instance (used to generate graph)

system('clear')

def Names(chainName="Node", chainLength=3, subScript=""):
	chainList = []
	for i in range(chainLength):
		chainList.append(chainName+subScript+str(i))	
	return chainList

def MarkovChain(chainList, markovOrder=1): #default uses 1st-order Markov assumption with 3 nodes in chain
	ebunch = []
	for n in range(len(chainList)-markovOrder):
		ebunch.append([chainList[n],chainList[n+markovOrder]])
	return ebunch

def InterProcess(causeChainList, effectChainList, markovLag=0): #markovLag=k creates cause[i] -> effect[i+k] otherwise cause[i] -> effect[i]
	ebunch = []
	if(len(causeChainList) != len(effectChainList)):
		print("Lists must be the same length!")
		pass
	else:
		for i in range(len(causeChainList)-markovLag):
			ebunch.append([causeChainList[i],effectChainList[i+markovLag]])
	return ebunch
	
def CommonPaths(nodeName, chainList, source=True): #source means it is a cause of all nodes in chain, false implies it is an effect of all the chain nodes
	ebunch = []
	if source:
		for link in chainList:
			ebunch.append([nodeName,link])
	else:
		for link in chainList:
			ebunch.append([link,nodeName])
	return ebunch	
	

