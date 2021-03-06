import torch.nn as nn
from torch.autograd import Variable
import torch

import scipy.sparse
import numpy as np
import graph

class ConvLSTMCell(nn.Module):

    def __init__(self, L,input_size, input_dim, hidden_dim, k):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.L=L
        self.length = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.K=k
        
        
        self.conv = GCLayer(L=self.L,Fin=self.input_dim + self.hidden_dim,
                              Fout=4 * self.hidden_dim,
                              K=self.K
                              )

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        h_cur=h_cur.float()
        c_cur=c_cur.float()
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.length)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.length)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, L, input_dim, hidden_dim, k, num_layers,
                 batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()

        #self._check_kernel_size_consistency(k)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        k = self._extend_for_multilayer(k, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(k) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_size= L.shape[0]
        self.L=L
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(L=self.L,input_size=self.input_size,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          k=self.k[i]
                                          ))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv0= GCLayer(L=self.L,Fin=sum(self.hidden_dim),
                              Fout=self.input_dim,
                              K=1
                              )

    
    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        


        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h,c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        
        #print (all_states.size())
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class GCLayer(nn.Module):
    def __init__(self,L, Fin, Fout, K):
        super(GCLayer, self).__init__()
        self.L=L
        self.Fin=Fin
        self.Fout=Fout
        self.K=K
        self.weigth=nn.Parameter(torch.ones(Fin*K,Fout)/(Fin*K*Fout))
        self.bais=nn.Parameter(torch.rand(1,Fout,1))
    
    def chebyshev5(self, x, L, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.
        
        Data: x of size N x F x M
            N: number of signals
            F: number of features per signal per vertex
            M: number of vertices
        """
        
        N, Fin, M = x.shape
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indice=torch.LongTensor(list(indices.T))
        data=torch.tensor(list(L.data))
        L = torch.sparse.FloatTensor(indice, data, L.shape)
        L=L.cuda()
        #L = tf.sparse_reorder(L)
        
        # Transform to Chebyshev basis
        x0 = x.permute(2, 1, 0)  # M x Fin x N

        x0 = x0.contiguous().view(M, -1)  # M x Fin*N
        x = x0.unsqueeze(0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
            return torch.cat([x, x_], 0)  # K x M x Fin*N
        if K > 1:
            x1 = torch.sparse.mm(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * torch.sparse.mm(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.view([K, M, Fin, N])  # K x M x Fin x N
        x = x.permute(3,1,2,0)  # N x M x Fin x K
        x = x.contiguous().view([N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        
        x = torch.mm(x, self.weigth)  # N*M x Fout
        x=x.contiguous().view(N, Fout, M) 
        x=x+self.bais
        return x  # N x Fout x M   
    
    def forward(self,x):
        x=self.chebyshev5(x,self.L,self.Fout,self.K)
        
        return x