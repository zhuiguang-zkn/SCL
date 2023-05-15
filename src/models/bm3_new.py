# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
paper:  Bootstrap Latent Representations for Multi-modal Recommendation
https://arxiv.org/abs/2207.05969
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import sys

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class TransformerFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super(TransformerFusion, self).__init__()
        # self.vision_dim = vision_dim
        # self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Vision and text embedding layers
        # self.vision_embedding = nn.Linear(vision_dim, hidden_dim)
        # self.text_embedding = nn.Linear(text_dim, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, vision_feats, text_feats):
        # Embed vision and text features
        # embedded_vision = self.vision_embedding(vision_feats)  # [batch_size, vision_dim] -> [batch_size, hidden_dim]
        # embedded_text = self.text_embedding(text_feats)  # [batch_size, text_dim] -> [batch_size, hidden_dim]

        # Concatenate the vision and text embeddings
        fused_feats = torch.cat((vision_feats, text_feats), dim=0)  # [2*batch_size, hidden_dim]

        # Transpose the fused features to [sequence_length, batch_size, hidden_dim]
        fused_feats = fused_feats.reshape(2, -1, self.hidden_dim)  # [2,  batch_size, hidden_dim]
        
        # print(fused_feats.shape, file=sys.stdout)

        # Pass the fused features through the Transformer encoder
        fused_feats = self.transformer_encoder(fused_feats)  # [2, batch_size, hidden_dim]

        # Squeeze the fused features to [batch_size, hidden_dim]
        fused_feats = fused_feats.transpose(0, 1)  # [batch_size, 2, hidden_dim] -> [batch_size, hidden_dim*2]
        
        # print(fused_feats.shape, file=sys.stdout)
        # Pass the fused features through an output layer
        fused_feats = self.output_layer(fused_feats.reshape(fused_feats.shape[0], -1))  # [batch_size, hidden_dim] -> [batch_size, hidden_dim]

        return fused_feats


# 生成online embedding 的时候， 额外加一个dropout操作
class BM3_NEW(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3_NEW, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.add_encoder = config['add_encoder']
        self.fusion_feature = config['fusion_feature']
        self.remove_v_t = config['remove_v_t']
            
        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.add_encoder:
            self.encoder = nn.Linear(self.embedding_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.encoder.weight)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)
        if self.v_feat is not None and self.t_feat is not None and self.fusion_feature:
            self.fusion_encoder = TransformerFusion(hidden_dim=self.embedding_dim, num_heads=1, num_layers=1)
            

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)
        if self.fusion_feature:
            f_feat_online = self.fusion_encoder(t_feat_online, v_feat_online)
            
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)
            if self.add_encoder:
                u_target = self.encoder(u_target)
                i_target = self.encoder(i_target)
            
            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)
                if self.add_encoder:
                    t_feat_target = self.encoder(t_feat_target)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)
                if self.add_encoder:
                    v_feat_target = self.encoder(v_feat_target)
            
            if self.fusion_feature:
                f_feat_target = f_feat_online.clone()
                f_feat_target = F.dropout(f_feat_target, self.dropout)
                

        if self.add_encoder:
            u_online, i_online = self.predictor(self.encoder(u_online_ori)), self.predictor(self.encoder(i_online_ori))
        else:
            u_online, i_online = self.predictor(F.dropout(u_online_ori)), self.predictor(F.dropout(i_online_ori))

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt, loss_f, loss_ft = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            if self.add_encoder:
                t_feat_online = self.predictor(self.encoder(t_feat_online))
            else:
                t_feat_online = self.predictor(F.dropout(t_feat_online))
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            if self.add_encoder:
                v_feat_online = self.predictor(self.encoder(v_feat_online))
            else:
                v_feat_online = self.predictor(F.dropout(v_feat_online))
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
        if self.fusion_feature:
            if self.add_encoder:
                f_feat_online = self.predictor(self.encoder(f_feat_online))
            else:
                f_feat_online = self.predictor(F.dropout(f_feat_online))
            f_feat_online = f_feat_online[items, :]
            f_feat_target = f_feat_target[items, :]
            loss_f = 1 - cosine_similarity(f_feat_online, i_target.detach(), dim=-1).mean()
            loss_ft = 1 - cosine_similarity(f_feat_online, f_feat_target.detach(), dim=-1).mean()
            

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        if self.remove_v_t:
            cl_loss = loss_f + loss_ft
        else:
            cl_loss = loss_t + loss_v + loss_tv + loss_vt + loss_f + loss_ft
        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (cl_loss).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

