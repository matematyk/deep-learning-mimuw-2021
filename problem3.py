import torch
import math 

class EmbeddingLayer(torch.nn.Module):
  def __init__(self, embedding_size: int, dictionary_size: int):
    super().__init__()
    self.embedding_size = embedding_size
    self.dictionary_size = dictionary_size

    self.dictionary = {i for  i in range (1,embedding_size)}

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      #todo liczby na liczby binarne
      Y = torch.zeros(self.embedding_size, x.shape[1], x.shape[0])
      for i in range(x.shape[0]):#batch_size
        seq = []
        for j in range(x.shape[1]): #seqence_length) [1,3,4,6]
          emb = [0]*self.embedding_size #[0,0,0,0,0,0,0]
          
          assert self.embedding_size > math.ceil(math.log(self.dictionary_size,2)) +1
          
          if int(x[i,j]) in self.dictionary: #czyli czy jest w alfabecie
            m = self.embedding_size - len(bin(x[i,j])[2:])
            wynik = (m-1)*"0" + bin(x[i,j])[2:] + "1"
            for key, value in enumerate(wynik): #nie moge odwracać
                #jesli nie jest odpowiedniej dlugosci 
                emb[key] = float(value)
            seq.append(emb)
            #Y[i,j,:] = emb #wrzuca pod batch_size
            
        Y[:,:,i] = torch.tensor(seq)
      return Y
    #(embedding_size, seqence_length, batch_size)



a = EmbeddingLayer(embedding_size=10, dictionary_size=10)
batch_size = 10
seqence_length = 10
data = [[1,1,1,3,4,5,6,1,1,1], [1,2,3,4,3,4,5,6,1,1]]
X = torch.tensor(data)
print(a.forward(X))