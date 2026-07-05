import torch 
import torch.nn 
import torch.nn.functional as F 


class NeuralNet(nn.Module):
  def __init__(self , n_input , n_output):
    super().__init__()
    self.L1 = nn.Linear(n_input , 128)
    self.L2 = nn.Linear(128 , 128)
    self.L3 = nn.Linear(128 , n_output)
  def forward(self , x):
    x = F.relu(self.L1(x))
    x = F.relu(self.L2(x))
    return self.L3(x)

