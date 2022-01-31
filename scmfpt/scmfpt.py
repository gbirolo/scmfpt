import numpy
#import pandas
#import matplotlib.pyplot as plt
#import seaborn
import torch
from torch import nn
from IPython import embed # FIXME remove when everything works

# TODO understand how do we handle device

def numpy2torch(array):
  return torch.tensor(array, dtype=torch.float32)#, device=dev)

class SNCMFModule(nn.Module):
  """Semi-Negative Convex Matrix Factorization"""
  def __init__(self, k, n_samples=None, n_feats=None, X=None, method='random'):
    super().__init__()
    if X is not None:
      assert n_samples is None or n_samples == X.shape[0]
      assert n_feats is None or n_feats == X.shape[1]
      n_samples, n_feats = X.shape
    comps = torch.rand((n_samples, k), dtype=torch.float32)#, device=dev)
    comps = comps/torch.sum(comps, dim=1).reshape((-1, 1))
    if method == 'random':
      profs = torch.normal(mean=0, std=1, size=(k, n_feats))
    elif method == 'random_samples':
      if X is None:
        raise ValueError(f"X argument require for method `random_sample`")
      from numpy.random import default_rng
      profs = numpy2torch(default_rng().choice(X, size=k, axis=0)) # FIXME use torch here
    else:
      raise ValueError("Unknown value '{method}' of method parameter")
    self.components = nn.Parameter(comps)
    self.profiles = nn.Parameter(profs)
  
  def forward(self, index=None):
    comps = self.components
    if index is not None:
      comps = comps[index]
    return torch.matmul(comps, self.profiles)
  def inner_loss(self):
    positive_loss = -torch.mean(torch.sum(torch.clamp(self.components, max=0)))
    convex_loss = torch.mean(torch.square(torch.sum(self.components, axis=1) - 1))
    return positive_loss + convex_loss

#  def to(self, device):
#    self.components = nn.Parameter(self.components.to(device))
#    self.profiles = nn.Parameter(self.profiles.to(device))
#    return self

class SCMF:
  def __init__(self, k, n_samples=None, n_feats=None, X=None, method='random', epochs=50000, optimizer=torch.optim.SGD, optimizer_kws={}):
    self.model = SNCMFModule(k, n_samples, n_feats, X, method)
    if 'lr' not in optimizer_kws:
      optimizer_kws['lr'] = 1
    self.optimizer = optimizer(self.model.parameters(), **optimizer_kws)
    self.epochs = epochs
    
  def fit(self, X, reconstruction_loss=nn.MSELoss()):
    X = torch.tensor(X, dtype=torch.float32) # FIXME should we add a parameter for the precision?
    self.model.train()
    for i in range(self.epochs):
      pred = self.model()
      train_loss = reconstruction_loss(pred, X) + self.model.inner_loss()
      
      self.optimizer.zero_grad()
      train_loss.backward()
      self.optimizer.step()

    with torch.no_grad():
      self.model.eval()
      return self.model.components.detach().numpy(), self.model.profiles.detach().numpy()
