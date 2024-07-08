import torch 

import torch.nn as nn

class StochasticCountModel(nn.Module):
    def __init__(self,no_of_groups: int, N: int, batch_size: int):
        super(StochasticCountModel, self).__init__()

        self.no_of_groups = no_of_groups

        self.N = N

        self.batch_size = batch_size

        self.countClass_hat = torch.ones((self.no_of_groups))
        self.countTotal_hat = torch.ones((self.no_of_groups))
        
        
        self.countClass_hat = self.countClass_hat*(self.N/(self.batch_size*self.no_of_groups)) 
        self.countTotal_hat = self.countTotal_hat*(self.N/self.batch_size) 
        
    def forward(self, rho: float, countClass_batch: int, countTotal_batch: int):
        self.countClass_hat = (1-rho)*self.countClass_hat + rho*(self.N/self.batch_size)*countClass_batch
        self.countTotal_hat = (1-rho)*self.countTotal_hat + rho*(self.N/self.batch_size)*countTotal_batch