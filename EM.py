import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from numpy.random import randint
from random import sample
from sklearn import metrics
from collections import Counter

Universal_Prob = []

def Plotear(M, S, titulo="default"):
	# add ellipse
	ax = plt.gca()
	for m, std in zip(M, S):
		plt.plot(m[0], m[1], ".")
		ellips = mpatches.Ellipse(m, 3*(2*std[0]), 3*(2*std[1]), angle=0, alpha=0.9, fill=False)
		ax.add_artist(ellips)

	# grid
	plt.grid(True,linestyle='--')

	# adjust the plotting aspect
	plt.gca().set_aspect('equal', adjustable='box')

	plt.title(titulo)
	plt.xlabel('X')
	plt.ylabel('Y')

def GetData(Clusters=2, pts=100):
	'''Generates non-linearly separable data'''
	print("######################## DATA ########################")
	X = []
	Y = []
	means = []
	stds = []

	minimo = 0
	for i in range(Clusters):
		mean = randint(minimo, 80, size=2)
		minimo = mean[0]
		std = randint(1, 25, size=2)
		cov = np.diag(np.square(std))

		x, y = np.random.multivariate_normal(mean, cov, pts).T #mean,covariance matrix
		
		X += x.tolist()
		Y += y.tolist()

		means.append(mean)
		stds.append(std)

		print("Distribution ", i+1)
		print("mean: ", mean)
		print("std: ", std)
		print("cov:\n", cov,"\n")

	label = []
	for i in range(Clusters):
		label += [i] * pts

	data = {'x': X, 'y': Y, 'label': label}

	#################### plot ######################
	plt.figure()
	plt.scatter(data['x'], data['y'], s=24, c=label)
	Plotear(means, stds, "DATA (Ground Truth)")
	################################################

	return data

def Performance(HipClust, TClust):
	FMI = metrics.fowlkes_mallows_score(TClust, HipClust)
	Rand_score = metrics.rand_score(TClust, HipClust)
	adjusted_Rand_score = metrics.adjusted_rand_score(TClust, HipClust)

	print("\n##################### Performance #####################")
	print("Confusion Matrix:")
	print(metrics.confusion_matrix(TClust, HipClust),"\n")

	print("Fowlkes-Mallows index (FMI): {:.2f}%".format(FMI * 100))
	print("Rand index: {:.2f}%".format(Rand_score * 100))
	print("Adjusted Rand index: {:.2f}%".format(adjusted_Rand_score * 100))
	print("########################################################\n")

def Initialize(data, clusters, points):
	'''ramdomly placed gaussians'''
	global Universal_Prob
	Universal_Prob = [[0.0]*points for _ in range(clusters)]

	mean = [[0.0, 0.0] for _ in range(clusters)]
	std = [[0.0, 0.0] for _ in range(clusters)]

	rand = sample(range(points), clusters) #random sampling without replacement
	for i in range(clusters):
		mean[i][0] = data['x'][rand[i]]
		mean[i][1] = data['y'][rand[i]]

		std[i][0] = randint(1, 5)
		std[i][1] = randint(1, 5)

	#sort means by the x-coordenate
	mean.sort(key=lambda m: m[0])

	#marginal probabiity for each cluster
	#could estimate priors as P(Ci) = P(Ci|x1) + P(Ci|x2) +...+ P(Ci|xp) / p
	#In this case, P(Ci) = 1 / Total number of clusters for all i = 1,2,...,n
	pm = [1.0/clusters for _ in range(clusters)]

	return {'M': mean, 'Std': std, 'P': pm}

def Prob_point(clusters, x1, x2, mean, std, pm):
	P = np.zeros(clusters)

	for c in range(clusters):
		#P[i] <- P(x)*P(Ci|x) = P(Ci)*P(x|Ci), the P(Ci)*P(x|Ci) is calculated here
		P[c] = pm[c] * norm.pdf(x1,mean[c][0],std[c][0]) * norm.pdf(x2,mean[c][1],std[c][1])

	return P.tolist()

def Expectation(Data, parameters, labels, clusters, points):
	global Universal_Prob

	convergence = True
	for i in range(points):
		x1 = Data['x'][i]
		x2 = Data['y'][i]

		Total_prob = Prob_point(clusters, x1, x2, parameters['M'], parameters['Std'], parameters['P'])
		Sum_TP = sum(Total_prob)

		if Sum_TP == 0:
			return convergence, True

		for c in range(clusters):
			#P(Ci|x) = P(Ci)*P(x|Ci)/P(x) where P(x) = P(C1)P(x|C1)+P(C2)P(x|C2)+...+P(Cn)P(x|Cn) = Sum_TP
			Universal_Prob[c][i] = Total_prob[c] / Sum_TP

		#Check if any sample changed of cluster
		predicted_cluster = np.argmax(Total_prob)
		if labels[i] != predicted_cluster:
			labels[i] = predicted_cluster
			convergence = False

	return convergence, False

def Maximization(Data, parameters, clusters, points):
	global Universal_Prob

	for c in range(clusters):
		# PClust <- P(Ci|x1) + P(Ci|x2) + P(Ci|x3) +...+ P(Ci|xp)
		PClust = sum(Universal_Prob[c])
		# update marginal probability for each cluster, P(Ci)
		parameters['P'][c] = PClust / points

		inv_P = 1.0 / PClust

		mean = [0.0, 0.0]
		for p in range(points):
			# mean <- P(Ci|x1)*x1 + P(Ci|x2)*x2 + P(Ci|x3)*x3 + ... + P(Ci|xp)*xp
			mean[0] += Universal_Prob[c][p] * Data['x'][p]
			mean[1] += Universal_Prob[c][p] * Data['y'][p]

		# mean = E[Ci] = [P(Ci|x1)*x1 + P(Ci|x2)*x2 +...+ P(Ci|xp)*xp] / [P(Ci|x1) + P(Ci|x2) +...+ P(Ci|xp)]
		mean = [i*inv_P for i in mean]
		# update mean for each cluster
		parameters['M'][c] = mean

		std = [0.0, 0.0]
		for p in range(points):
			# std(var) <- P(Ci|x1)*(x1-mean)^2 + P(Ci|x2)*(x2-mean)^2 + ... + P(Ci|xp)*(xp-mean)^2
			std[0] += Universal_Prob[c][p] * ((Data['x'][p] - mean[0])**2)
			std[1] += Universal_Prob[c][p] * ((Data['y'][p] - mean[1])**2)

		# var = E[var] = [P(Ci|x1)*(x1-mean)^2 + P(Ci|x2)*(x2-mean)^2 + ... + P(Ci|xp)*(xp-mean)^2] / [P(Ci|x1) + P(Ci|x2) +...+ P(Ci|xp)]
		# std <- sqrt( var )
		# update std deviation for each cluster
		parameters['Std'][c] = [(i*inv_P)**(0.5) for i in std]

	return parameters

def EM(Data, total_clusters, points):
	print("\n######### Expectation Maximization Algorithm #########")
	predicted_clusters = list(range(points))

	restart = True
	while restart:
		#start with randomly placed Gaussians
		parameters = Initialize(Data, total_clusters, points)
		p = copy.deepcopy(parameters)
		
		convergence = False
		count = 0
		while not convergence:
			### Expectation-STEP ###
			convergence, restart = Expectation(Data, parameters, predicted_clusters, total_clusters, points)

			if restart:
				print("Error! Total probability equal zero. Restart necessary.")
				break

			### Maximization-STEP ###
			parameters = Maximization(Data, parameters, total_clusters, points)

			count += 1
			print("Iterations: ", count)

	#Points per cluster
	print("\nPoints per Cluster: ")
	PpC = Counter(predicted_clusters)
	for k in PpC:
		print("cluster ",k,"= ", PpC[k])

	################################ PLOT
	plt.figure()
	plt.scatter(Data['x'], Data['y'], s=24, c='tab:gray')
	Plotear(p['M'], p['Std'], "INITIALIZATION")
	################################ PLOT
	plt.figure()
	plt.scatter(Data['x'], Data['y'], s=24, c=predicted_clusters)
	Plotear(parameters['M'], parameters['Std'], "PREDICTED CLUSTERS")
	#################################

	return predicted_clusters


if __name__ == '__main__':
	total_clusters = 2  #number of clusters

	Data = GetData(Clusters=total_clusters, pts=150) #clusters,points per cluster
	points = len(Data['x']) #Total points
	print("Total Data Points: ", points)

	predicted_clusters = EM(Data, total_clusters, points)

	Performance(predicted_clusters, Data['label'])

	plt.show()