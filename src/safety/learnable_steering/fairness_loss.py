import numpy as np

import torch

def computeBatchCounts(protectedAttributes,intersectGroups,predictions):
    # intersectGroups should be pre-defined so that stochastic update of p(y|S) 
    # can be maintained correctly among different batches   
     
    # compute counts for each intersectional group
    countsClassOne = torch.zeros((len(intersectGroups)),dtype=torch.float)
    countsTotal = torch.zeros((len(intersectGroups)),dtype=torch.float)
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] = countsTotal[index] + 1
        countsClassOne[index] = countsClassOne[index] + predictions[i]        
    return countsClassOne, countsTotal

def differentialFairnessBinaryOutcomeTrain(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerGroup = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float)
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0) # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = torch.max(epsilon,torch.abs(torch.log(probabilitiesOfPositive[i])-torch.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[i]))-(torch.log(1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = torch.max(epsilonPerGroup) # overall DF of the algorithm 
    return epsilon

def fairness_loss(base_fairness,stochasticModel):
    # DF-based penalty term
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    zeroTerm = torch.tensor(0.0) 
    
    theta = (stochasticModel.countClass_hat + dirichletAlpha) /(stochasticModel.countTotal_hat + concentrationParameter)
    epsilonClass = differentialFairnessBinaryOutcomeTrain(theta)
    return torch.max(zeroTerm, (epsilonClass-base_fairness))