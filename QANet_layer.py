import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from util import masked_softmax

def clones( module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


class QANetHighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(QANetHighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class QANetEmbedding(nn.Module):
    """Embedding layer in QANet
    Concatenate word embedding and char-level CNN embedding to get word representation.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor):Initial char vectors.
        hidden_size (int): Size of hidden activations after concatenation.
        for word embedding and char-level CNN, we assign H/2 for each
        char_len: max length of char in word
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, char_vectors,hidden_size,char_len, drop_prob):
        super(QANetEmbedding, self).__init__()
        assert hidden_size%2==0
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed=nn.Embedding.from_pretrained(char_vectors,freeze=False)
        self.word_proj = nn.Linear(word_vectors.size(1), hidden_size//2, bias=False)
        self.char_len=char_len
        self.char_embed_dim=char_vectors.size(1)
        self.char_CNN=nn.Sequential(nn.Conv1d(self.char_embed_dim,hidden_size//2,3,padding=1),
                                     nn.BatchNorm1d(hidden_size//2),
                                     nn.ReLU(),
                                     nn.MaxPool1d(self.char_len),
                                     nn.Dropout(self.drop_prob))
        self.hwy = QANetHighwayEncoder(2, hidden_size)

    def forward(self, word_x,char_x):
        word_emb = self.word_embed(word_x)  # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = self.word_proj(word_emb)  # (batch_size, seq_len, hidden_size)

        char_emb=self.char_embed(char_x)#(batch_size,seq_len, word_len,embed_size)
        batch_size=char_x.size(0)
        char_emb=char_emb.view(-1,self.char_len,self.char_embed_dim).transpose(-1,-2).contiguous()
        char_emb=self.char_CNN(char_emb)
        hidden_size=char_emb.size(-2)
        char_emb=char_emb.view(batch_size,-1,hidden_size)
        final_emb=torch.cat([char_emb,word_emb],dim=-1)
        emb = self.hwy(final_emb)  # (batch_size, seq_len, hidden_size)

        return emb


class PositionalEncoding(nn.Module):
    '''
    "Implement the PE function."
    Args
        d_model(int): hidden_size of encoder and decoder
    '''


    def __init__(self, d_model, drop_prob, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class InitializedConv1d(nn.Module):
    """
    Replace default linear projection
    Args:
        in_channels(int): input size
        out_channels(int): output size
    """
    def __init__(self, in_channels, out_channels, kernel_size=1,groups=1,padding=0,stride=1,relu=False,bias=False):
        super(InitializedConv1d, self).__init__()
        self.conv1d=nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             groups=groups,
                             padding=padding,
                             stride=stride,
                              bias=bias)
        if relu:
            self.relu=True
            nn.init.kaiming_normal_(self.conv1d.weight,nonlinearity="relu")
        else:
            self.relu=False
            nn.init.xavier_normal_(self.conv1d.weight)

    def forward(self,x):
        x = x.transpose(-1, -2).contiguous()
        if self.relu:
            x=F.relu(self.conv1d(x))

        else:
            x=self.conv1d(x)
        return x.transpose(-1,-2).contiguous()

def attention(query,key,value,mask,dropout=None):
    """
    self attention function with masking

    """
    d_k=query.size(-1)
    score=torch.matmul(query,key.transpose(-1,-2).contiguous())/math.log(d_k) # Batch_size * h * seq_len * seq_len
    score=score.masked_fill_(mask==0,-1e30)
    attn=F.softmax(score,dim=-1)
    if dropout is not None:
        attn=dropout(attn)
    return torch.matmul(attn,value), attn  # Batch_size * h * seq_len * d_k

class MultiHeadedAttention(nn.Module):
    """
    Multi-Header attention mechanism
    Args:
        hidden_size(int): hidden size of model
        h(int):number of head in multi-head attention
    """
    def __init__(self,hidden_size,h,drop_prob=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size%h==0
        self.hidden_size=hidden_size
        self.h=h
        self.attn=None
        self.linears=clones(InitializedConv1d(hidden_size,hidden_size),4)
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,query,key,value,mask):
        batch_size=query.size(0)
        mask=mask.view(batch_size,1,1,-1)
        d_k=self.hidden_size//self.h
        query,key,value=[l(x).view(batch_size,-1,self.h,d_k).transpose(1,2)
                         for x,l in zip((query,key,value),self.linears)]   # Batch_size * h * seq_len * d_k

        x, self.attn=attention(query,key,value,mask)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * d_k)  # Batch_size * seq_len * hidden_size
        return self.linears[-1](x)

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution
    Args:
        in_channels: number of channels in the input, here is the hidden size of encoder
        out_channels:number of channels in the output, here is also the hidden size of encoder
    """
    def __init__(self, in_channels,out_channels,kernel_size,padding=0, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.conv1=nn.Conv1d(in_channels=in_channels,
                             out_channels=in_channels,
                             kernel_size=kernel_size,
                             groups=in_channels,
                             padding=kernel_size//2,
                             bias=False)
        self.conv2=nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             groups=1,
                             padding=padding,
                             bias=bias)

    def forward(self,x):
        x = x.transpose(-1, -2).contiguous()
        x = F.relu(self.conv2(self.conv1(x)))
        return x.transpose(-1,-2).contiguous()

class PointwiseFeedForwardNetwork(nn.Module):
    """
    Feed forward NN
    """
    def __init__(self,hidden_size):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.linear1=InitializedConv1d(hidden_size,hidden_size)
        self.linear2=InitializedConv1d(hidden_size,hidden_size)
    def forward(self,x):
        return F.relu(self.linear2(self.linear1(x)))


def layerDropout(self,x,res, drop_prob):
    """
    Implement layer dropout
    """
    if self.training:
        pred = torch.empty(1).uniform_(0, 1) < drop_prob
        if pred:
            return x
        else:
            return x+res
    else:
        return x+res

class SubDropoutLayerConnection(nn.Module):
    def __init__(self,hidden_size,drop_prob=0.1):
        super(SubDropoutLayerConnection, self).__init__()
        self.norm=nn.LayerNorm(hidden_size)
        self.dropout=nn.Dropout(drop_prob)
        self.drop_prob=drop_prob

    def forward(self,x,sublayer,position):
        res=self.dropout(sublayer(self.norm(x)))
        layer_drop_prob=self.drop_prob*position
        return layerDropout(self,x,res,layer_drop_prob)

class QANetEncoder(nn.Module):
    def __init__(self,hidden_size,attn,conv,feed_forward,conv_num,drop_prob=0.1):
        super(QANetEncoder, self).__init__()
        self.hidden_size=hidden_size
        self.attn=attn
        self.conv_list=clones(conv,conv_num)
        self.feed_forward=feed_forward
        self.pe=PositionalEncoding(hidden_size,drop_prob)
        self.L=conv_num+2
        self.sublayer_list=clones(SubDropoutLayerConnection(hidden_size,drop_prob),self.L)
        self.drop_prob=drop_prob


    def forward(self,x,mask):
      #  x=self.pe(x)
        for i,l in enumerate(self.conv_list):
            x=self.sublayer_list[i](x,l,(i+1)/self.L)

        x=self.sublayer_list[-2](x,lambda x:self.attn(x,x,x,mask),(self.L-1)/self.L)
        x=self.sublayer_list[-1](x,self.feed_forward,1)
        return x


class QANetEncoderBlock(nn.Module):
    def __init__(self,layer,N):
        super(QANetEncoderBlock, self).__init__()
        self.layer_list=clones(layer,N)
        self.norm=nn.LayerNorm(layer.hidden_size)

    def forward(self,x,mask):
        for l in self.layer_list:
            x=l(x,mask)
        return self.norm(x)

class QANetAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF and also default to QANet.

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(QANetAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s



class QANetOutput(nn.Module):
    def __init__(self,hidden_size):
        super(QANetOutput, self).__init__()
        self.start_linear1=InitializedConv1d(hidden_size,1)
        self.start_linear2=InitializedConv1d(hidden_size,1)
        self.end_linear1=InitializedConv1d(hidden_size,1)
        self.end_linear2=InitializedConv1d(hidden_size,1)

    def forward(self,m0,m1,m2,mask):
        p1=self.start_linear1(m0)+self.start_linear2(m1)
        p2=self.end_linear1(m0)+self.end_linear2(m2)

        log_p1=masked_softmax(p1.squeeze(),mask,log_softmax=True)
        log_p2=masked_softmax(p2.squeeze(),mask,log_softmax=True) 

        return log_p1,log_p2