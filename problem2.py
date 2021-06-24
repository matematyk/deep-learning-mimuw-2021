import torch
import torch.nn as nn
from numpy.random import randn

BATCH_SIZE = 20
class RecurrentLayer(torch.nn.Module):
  def __init__(self, input_size: int, hidden_size: int, p_dropout: float):
    super(RecurrentLayer, self).__init__()
    self.output_size = 67
    self.hidden_size = hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(hidden_size, self.output_size)
    self.hidden = torch.zeros(self.hidden_size, BATCH_SIZE)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()


  def forward(self, x): #(input_size, seqence_length, batch_size),
    Y = torch.zeros([self.output_size, x.shape[1], x.shape[2]])
    for j in range(x.shape[2]):
      drop = torch.full([self.hidden_size], 0.5)
      drop = torch.bernoulli(drop)
      self.hidden[:,j] = self.hidden[:,j] * drop 
      #* pointwise, @ mnozenie macierzy
    for i in range(x.shape[1]):
      combined = torch.cat((torch.squeeze(x[:,i,:]), self.hidden), 0)
      combined = combined.T #(batch_size, input_size)
      self.hidden = self.tanh(self.i2h(combined))
      output = self.i2o(self.hidden)
      self.hidden = self.hidden.T
      output = self.relu(output)
      Y[:,i,:] = output.T
      

    return Y

n_words=57
seqence_length=10
batch_size =2

rnn = RecurrentLayer(input_size=n_words, hidden_size=128, p_dropout=0.4)

seqence_length = 6 #zbior wektorów z R^n, input_size, seqence_length
# (input_size, seqence_length, batch_size) #Adam -> 'A', 'd', 'a', 'm' -> Polskie czy Angielskie imie
x = torch.rand([n_words, seqence_length, BATCH_SIZE])
#x = torch.rand([n_letters, seqence_length, batch_size])
y = rnn.forward(x) #wybieramy 1 literke! # 'I' 'am' a boss -> '
print(y.shape)