"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import BiDAF_layer
import QANet_layer
import KnowGQA_layer
import torch
import torch.nn as nn
import copy


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = BiDAF_layer.BiDAFEmbedding(word_vectors=word_vectors,
                                         hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.enc = BiDAF_layer.BiDAFRNNEncoder(input_size=hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=1,
                                          drop_prob=drop_prob)

        self.att = BiDAF_layer.BiDAFAttention(hidden_size=2 * hidden_size,
                                              drop_prob=drop_prob)

        self.mod = BiDAF_layer.BiDAFRNNEncoder(input_size=8 * hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=2,
                                          drop_prob=drop_prob)

        self.out = BiDAF_layer.BiDAFOutput(hidden_size=hidden_size,
                                           drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out





class BiDAF_char(nn.Module):
    """Default BiDAF model(add char embedding)

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors(torch.Tensor): Initial Char vectors.
        Char_len(int): The maximum length of char in each word.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, char_len, hidden_size, drop_prob=0.):
        super(BiDAF_char, self).__init__()
        self.emb = BiDAF_layer.BiDAFEmbeddingwithChar(word_vectors=word_vectors,
                                                   char_vectors=char_vectors,
                                                   hidden_size=hidden_size,
                                                   char_len=char_len,
                                                   drop_prob=drop_prob)

        self.enc = BiDAF_layer.BiDAFRNNEncoder(input_size=hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=1,
                                          drop_prob=drop_prob)

        self.att = BiDAF_layer.BiDAFAttention(hidden_size=2 * hidden_size,
                                              drop_prob=drop_prob)

        self.mod = BiDAF_layer.BiDAFRNNEncoder(input_size=8 * hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=2,
                                          drop_prob=drop_prob)

        self.out = BiDAF_layer.BiDAFOutput(hidden_size=hidden_size,
                                           drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs,cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs,qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out



class  QANet(nn.Module):
    """
    Default  QANet implementation

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors(torch.Tensor): Initial Char vectors.
        Char_len(int): The maximum length of char in each word.
        hidden_size (int): Number of features in the hidden state at each layer.
        h(int):Number of header in each self attention layer.
        drop_prob (float): Dropout probability.

    """
    def __init__(self, word_vectors, char_vectors, char_len,hidden_size=512,h=4, drop_prob=0.1):
        super(QANet, self).__init__()
        c = copy.deepcopy
        self.hidden_size=hidden_size
        self.emb = QANet_layer.QANetEmbedding(word_vectors=word_vectors,
                                              char_vectors=char_vectors,
                                              hidden_size=hidden_size,
                                              char_len=char_len,
                                              drop_prob=drop_prob)

        attn=QANet_layer.MultiHeadedAttention(hidden_size=hidden_size,
                                              h=h,
                                              drop_prob=drop_prob)

        feed_forward=QANet_layer.PointwiseFeedForwardNetwork(hidden_size=hidden_size)

        emb_conv=QANet_layer.DepthwiseSeparableConv1d(in_channels=hidden_size,
                                                     out_channels=hidden_size,
                                                     kernel_size=7)

        model_conv=QANet_layer.DepthwiseSeparableConv1d(in_channels=hidden_size,
                                                     out_channels=hidden_size,
                                                     kernel_size=5)
        embed_encoder=QANet_layer.QANetEncoder(hidden_size=hidden_size,
                                               attn=c(attn),
                                               feed_forward=c(feed_forward),
                                               conv=c(emb_conv),
                                               conv_num=4,
                                               drop_prob=drop_prob)

        self.c_emb_encoder=QANet_layer.QANetEncoderBlock(c(embed_encoder),1)
        self.q_emb_encoder=QANet_layer.QANetEncoderBlock(c(embed_encoder),1)



        self.att = QANet_layer.QANetAttention(hidden_size=hidden_size,
                                              drop_prob=drop_prob)

        self.att_porj=QANet_layer.InitializedConv1d(4*hidden_size,hidden_size)


        model_encoder=QANet_layer.QANetEncoder( hidden_size=hidden_size,
                                                attn=c(attn),
                                                feed_forward=c(feed_forward),
                                                conv=c(model_conv),
                                                conv_num=2,
                                                drop_prob=drop_prob)
        self.modeling_encoder=QANet_layer.QANetEncoderBlock(c(model_encoder),7)

        self.out_gen=QANet_layer.QANetOutput(hidden_size=hidden_size)



    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs,cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs,qc_idxs)         # (batch_size, q_len, hidden_size)

        q_syntac_emb=self.q_emb_encoder(q_emb,q_mask)
        c_syntac_emb=self.c_emb_encoder(c_emb,c_mask)

        att = self.att(c_syntac_emb, q_syntac_emb,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        att_proj=self.att_porj(att)

        m0=self.modeling_encoder(att_proj,c_mask)
        m1=self.modeling_encoder(m0,c_mask)
        m2=self.modeling_encoder(m1,c_mask)

        out=self.out_gen(m0,m1,m2,c_mask)

        return out



class KnowGQA(nn.Module):
    '''
    KnowGQA model
    Since KnowGQA share many strucutre from QANet, we directly use the layer in QANet_layer.py
    Args
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors(torch.Tensor): Initial Char vectors.
        Char_len(int): The maximum length of char in each word.
        hidden_size (int): Number of features in the hidden state at each layer.
        h(int):Number of header in each self attention layer.
        drop_prob (float): Dropout probability.
    '''
    def __init__(self, word_vectors, char_vectors, char_len,hidden_size=512,h=4, drop_prob=0.1):
        super(KnowGQA, self).__init__()
        c = copy.deepcopy
        self.hidden_size=hidden_size
        embedding_size=word_vectors.size(1)
        self.emb = QANet_layer.QANetEmbedding(word_vectors=word_vectors,
                                              char_vectors=char_vectors,
                                              hidden_size=hidden_size,
                                              char_len=char_len,
                                              drop_prob=drop_prob)

        attn=QANet_layer.MultiHeadedAttention(hidden_size=hidden_size,
                                              h=h,
                                              drop_prob=drop_prob)

        feed_forward=QANet_layer.PointwiseFeedForwardNetwork(hidden_size=hidden_size)

        emb_conv=QANet_layer.DepthwiseSeparableConv1d(in_channels=hidden_size,
                                                     out_channels=hidden_size,
                                                     kernel_size=7)

        model_conv=QANet_layer.DepthwiseSeparableConv1d(in_channels=hidden_size,
                                                     out_channels=hidden_size,
                                                     kernel_size=5)
        embed_encoder=QANet_layer.QANetEncoder(hidden_size=hidden_size,
                                               attn=c(attn),
                                               feed_forward=c(feed_forward),
                                               conv=c(emb_conv),
                                               conv_num=4,
                                               drop_prob=drop_prob)

        self.c_emb_encoder=QANet_layer.QANetEncoderBlock(c(embed_encoder),1)
        self.q_emb_encoder=QANet_layer.QANetEncoderBlock(c(embed_encoder),1)



        self.att = QANet_layer.QANetAttention(hidden_size=hidden_size,
                                              drop_prob=drop_prob)

        self.GCNBlock=KnowGQA_layer.KnowledgeRepresentation(input_size=embedding_size,

                                                           output_size=hidden_size,
                                                            N=3,
                                                            drop_prob=drop_prob)

        self.knowledge_att=KnowGQA_layer.KnowledgeAttention_s(hidden_size=hidden_size)

        self.att_porj=QANet_layer.InitializedConv1d(4*hidden_size,hidden_size)


        model_encoder=QANet_layer.QANetEncoder( hidden_size=hidden_size,
                                                attn=c(attn),
                                                feed_forward=c(feed_forward),
                                                conv=c(model_conv),
                                                conv_num=2,
                                                drop_prob=drop_prob)
        self.modeling_encoder=QANet_layer.QANetEncoderBlock(c(model_encoder),7)

        self.out_gen=QANet_layer.QANetOutput(hidden_size=hidden_size)



    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs,co_idxs,adjs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cw_idxs,cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs,qc_idxs)         # (batch_size, q_len, hidden_size)

        q_syntac_emb=self.q_emb_encoder(q_emb,q_mask)
        c_syntac_emb=self.c_emb_encoder(c_emb,c_mask)

        co_emb=self.emb.word_embed(co_idxs)    #(batch_size, g_len, hidden_size)
        g=self.GCNBlock(co_emb,adjs)           #(batch_size, hidden_size)

        q_g=self.knowledge_att(q_syntac_emb,g,q_mask)     # (batch_size, q_len, hidden_size)
        c_g=self.knowledge_att(c_syntac_emb,g,c_mask)     #  (Batch_size, c_len, hidden_size)

        att = self.att(c_g, q_g,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * hidden_size)

        att_proj=self.att_porj(att)         #(batch_size, c_len, hidden_size)


        m0=self.modeling_encoder(att_proj,c_mask)        #(batch_size, c_len, hidden_size)
        m1=self.modeling_encoder(m0,c_mask)          #(batch_size, c_len, hidden_size)
        m2=self.modeling_encoder(m1,c_mask)          #(batch_size, c_len, hidden_size)

        out=self.out_gen(m0,m1,m2,c_mask)

        return out
