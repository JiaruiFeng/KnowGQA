import torch
import torch.nn as nn
import torch.nn.functional as F
from QANet_layer import clones,InitializedConv1d
from util import masked_softmax
class GCN(nn.Module):
    '''
    Graph convolutional Network layer
    Args:
        input_size(int): the hidden length of input word embdedding
        output_size(int): the output length of word embedding
    '''
    def __init__(self,input_size,output_size,drop_prob=0.1):
        super(GCN, self).__init__()
        self.proj=InitializedConv1d(input_size,output_size)
        self.dropout=nn.Dropout(drop_prob)
        self.norm=nn.LayerNorm(output_size)
    def forward(self,x,adj):
        x=F.relu(self.dropout(self.proj(torch.matmul(adj,x))))
        return self.norm(x)

class Diffpool(nn.Module):
    '''
    Diffpool layer, which can generate compact graph representation.
    Here, we simultaneously compact both graph size and hidden size.
    Args:
        input_size(int): the hidden length of input word embedding
        output_size(int): The hidden length of output word embedding, and also the output graph size.
    '''
    def __init__(self,input_size,output_size,drop_prob=0.1):
        super(Diffpool, self).__init__()
        self.repre=GCN(input_size,output_size,drop_prob)
        self.embed=GCN(input_size,output_size,drop_prob)

    def forward(self,x,adj):
        z=self.repre(x,adj)
        s=self.embed(x,adj)
        s=F.softmax(s,dim=-1)
        x=torch.matmul(s.transpose(-1,-2),z)
        adj=torch.matmul(torch.matmul(s.transpose(-1,-2),adj),s)
        return x,adj


class KnowledgeRepresentation(nn.Module):
    '''
    Residual GCN Block
    Args:
        input_size(int): the hidden length of input word embedding
        output_size(int): The hidden length of output word embedding, and also the output graph size.

    '''
    def __init__(self,input_size,output_size,N,drop_prob=0.1):
        super(KnowledgeRepresentation, self).__init__()
        self.diffpool=Diffpool(input_size,output_size,drop_prob)
        self.GCN_layers=clones(GCN(output_size,output_size,drop_prob),N-1)

    def forward(self,x,adj):
        x,adj=self.diffpool(x,adj)   # Batch_size * output_size * output_size
        for l in self.GCN_layers:
            x=l(x,adj)+x             # Batch_size * output_size * output_size

        return F.relu(x.sum(dim=-2))   # Batch_size * output_size

'''
class KnowledgeAttention(nn.Module):

    def __init__(self,hidden_size):
        super(KnowledgeAttention, self).__init__()
        self.gate_proj=InitializedConv1d(4*hidden_size,1,bias=True)
        assert hidden_size%2==0
        self.proj1=nn.Linear(hidden_size,hidden_size//2)
        self.proj2=nn.Linear(hidden_size,hidden_size//2)
    def forward(self,m,u,g,mask):
        g=g.unsqueeze(-1)
        knowledge_att=torch.matmul(m,g)    #Batch_size * c_len * 1
        batch_size=knowledge_att.size(0)
        mask=mask.view(batch_size,-1,1)
        knowledge_att=masked_softmax(knowledge_att,mask,dim=-2)


        m_g=m.transpose(-1,-2)        #Batch_size * hidden_size *c_len
        u_g=u.transpose(-1,-2)        #Batch_size * hidden_size *c_len

        m_g=F.relu(self.proj1(torch.matmul(m_g, knowledge_att).squeeze()))         #Batch_size * hidden_size//2
        u_g=F.relu(self.proj2(torch.matmul(u_g,knowledge_att).squeeze()))     #Batch_size * hidden_size//2


        u_g=torch.cat([u_g,m_g],dim=-1)        #Batch_size * hidden_size
        u_g=u_g.unsqueeze(-2)       #Batch_size * 1 *hidden_size

        c_len=m.size(-2)
        U_g=u_g.repeat((1,c_len,1))         #Batch_size * c_len * hidden_size

        #gate function
        gate=torch.cat([u,U_g,torch.mul(u,U_g),torch.sub(u,U_g)],dim=-1)  #Batch_size * c_len *4*hidden_size
        gate=torch.sigmoid(self.gate_proj(gate))
        M_g=torch.mul(gate,u)+torch.mul((1-gate),U_g)
        return M_g


'''


class KnowledgeAttention_s(nn.Module):
    '''
    Knowledge attention mechanism
    '''
    def __init__(self,hidden_size):
        super(KnowledgeAttention_s, self).__init__()
        self.gate_proj=InitializedConv1d(4*hidden_size,1,bias=True)
        assert hidden_size%2==0
    def forward(self,m,g,mask):
        g=g.unsqueeze(-1)
        knowledge_att=torch.matmul(m,g)    #Batch_size * c_len * 1
        batch_size=knowledge_att.size(0)
        mask=mask.view(batch_size,-1,1)
        knowledge_att=masked_softmax(knowledge_att,mask,dim=-2)

        m_g=m.transpose(-1,-2)        #Batch_size * hidden_size *c_len

        m_g=F.relu(torch.matmul(m_g, knowledge_att).squeeze())         #Batch_size * hidden_size

        m_g=m_g.unsqueeze(-2)       #Batch_size * 1 *hidden_size

        c_len=m.size(-2)
        M_g=m_g.repeat((1,c_len,1))         #Batch_size * c_len * hidden_size

        #gate function
        gate=torch.cat([m,M_g,torch.mul(m,M_g),torch.sub(m,M_g)],dim=-1)  #Batch_size * c_len *4*hidden_size
        gate=torch.sigmoid(self.gate_proj(gate))
        M_g=torch.mul(gate,m)+torch.mul((1-gate),M_g)    # Batch_size * c_len * hidden_size
        return M_g
