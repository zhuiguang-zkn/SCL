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

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class Simsiam(GeneralRecommender):
    def __init__(self, config, dataset):
        super(Simsiam, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.intra_weight = config['intra_weight']
        # self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.config = config

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        # maintain a unique predictor for each modality(include user, item, image, text)
        self.user_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.item_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.image_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.text_predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.reg_loss = EmbLoss()

        # nn.init.xavier_normal_(self.predictor.weight)
        nn.init.xavier_normal_(self.item_predictor.weight)
        nn.init.xavier_normal_(self.user_predictor.weight)
        nn.init.xavier_normal_(self.image_predictor.weight)
        nn.init.xavier_normal_(self.text_predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

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
    def calculate_distance(self, p, z):
        z = z.detach()
        return 1-cosine_similarity(p, z.detach(), dim=-1).mean()
    
    def calculate_loss(self, interactions):
        # use simsam to calculate loss
        # loss contain inter-modal and intra-moldal loss
        l_intra, l_inter = 0.0, 0.0
        u_ori, i_ori = self.forward()
        if self.t_feat is not None:
            t_feat = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat = self.image_trs(self.image_embedding.weight)
        
        users, items = interactions[0], interactions[1]
        u_ori = u_ori[users, :]
        i_ori = i_ori[items, :]
        if self.t_feat is not None:
            t_feat = t_feat[items, :]
        if self.v_feat is not None:
            v_feat = v_feat[items, :]
        
        
        u_1 , u_2 = F.dropout(u_ori, self.dropout), F.dropout(u_ori, self.dropout)
        i_1, i_2 = F.dropout(i_ori, self.dropout), F.dropout(i_ori, self.dropout)
        assert (u_1!=u_2).any() and (i_1!=i_2).any(), (u_1, u_2)
        
        if self.v_feat is not None:
            v_1, v_2 = F.dropout(v_feat, self.dropout), F.dropout(v_feat, self.dropout)
            assert (v_1!=v_2).any() 
        if self.t_feat is not None:
            t_1, t_2 = F.dropout(t_feat, self.dropout), F.dropout(t_feat, self.dropout)
            assert (t_1!=t_2).any()
            
        if self.config['one_predictor']:
            u_1_pred, u_2_pred = self.predictor(u_1), self.predictor(u_2)
            i_1_pred, i_2_pred = self.predictor(i_1), self.predictor(i_2)
            if self.v_feat is not None:
                v_1_pred, v_2_pred = self.predictor(v_1), self.predictor(v_2)
            if self.t_feat is not None:
                t_1_pred, t_2_pred = self.predictor(t_1), self.predictor(t_2)
            
        else:
            u_1_pred, u_2_pred = self.user_predictor(u_1), self.user_predictor(u_2)
            i_1_pred, i_2_pred = self.item_predictor(i_1), self.item_predictor(i_2)
            if self.v_feat is not None:
                v_1_pred, v_2_pred = self.image_predictor(v_1), self.image_predictor(v_2)
            if self.t_feat is not None:
                t_1_pred, t_2_pred = self.text_predictor(t_1), self.text_predictor(t_2)
            
        # intra-modal loss
        u_u_loss = self.calculate_distance(u_1_pred, u_2) + self.calculate_distance(u_2_pred, u_1)
        l_intra += u_u_loss
        i_i_loss = self.calculate_distance(i_1_pred, i_2) + self.calculate_distance(i_2_pred, i_1)
        l_intra += i_i_loss
        if self.t_feat is not None:
            t_t_loss = self.calculate_distance(t_1_pred, t_2) + self.calculate_distance(t_2_pred, t_1)
            l_intra += t_t_loss
        if self.v_feat is not None:
            v_v_loss = self.calculate_distance(v_1_pred, v_2) + self.calculate_distance(v_2_pred, v_1)
            l_intra += v_v_loss
            
        # print("-----------------------------")
        # print(f"u_u_loss: {u_u_loss}, i_i_loss:{i_i_loss}, t_t_loss:{t_t_loss}, v_v_loss:{v_v_loss}")
        # print("-----------------------------")
            
            
        # inter-modal loss    
        if self.config['one_predictor']:
            u_pred = self.predictor(u_ori)
            i_pred = self.predictor(i_ori)
            if self.t_feat is not None:
                t_pred = self.predictor(t_feat)
            if self.v_feat is not None:
                v_pred = self.predictor(v_feat)
        else:
            u_pred = self.user_predictor(u_ori)
            i_pred = self.item_predictor(i_ori)
            if self.t_feat is not None:
                t_pred = self.text_predictor(t_feat)
            if self.v_feat is not None:
                v_pred = self.image_predictor(v_feat)
        
        u_i_loss = self.calculate_distance(u_pred, i_ori) + self.calculate_distance(i_pred, u_ori)
        l_inter += u_i_loss
        if self.t_feat is not None:
            i_t_loss = self.calculate_distance(i_pred, t_feat) + self.calculate_distance(t_pred, i_ori)
            l_inter += i_t_loss
        if self.v_feat is not None:
            i_v_loss = self.calculate_distance(i_pred, v_feat) + self.calculate_distance(v_pred, i_ori)
            l_inter += i_v_loss
        if self.t_feat is not None and self.v_feat is not None:
            t_v_loss = self.calculate_distance(t_pred, v_feat) + self.calculate_distance(v_pred, t_feat)
            # l_inter += t_v_loss   # don't add this loss

        # print("-----------------------------")
        # print(f"u_i_loss: {u_i_loss}, i_t_loss:{i_t_loss}, i_v_loss:{i_v_loss}")
        # print("-----------------------------")
        if self.config['return_dict_loss']:
            return {
                "loss": l_inter + self.intra_weight * l_intra + self.reg_weight * self.reg_loss(u_ori, i_ori),
                "u_u_loss": u_u_loss,
                "i_i_loss": i_i_loss,
                "v_v_loss": v_v_loss,
                "t_t_loss": t_t_loss,
                "u_i_loss": u_i_loss,
                "i_t_loss": i_t_loss,
                "i_v_loss": i_v_loss
            }

        return l_inter + self.intra_weight * l_intra + self.reg_weight * self.reg_loss(u_ori, i_ori)
                        
                
            
        
        # u_ori, i_ori = self.forward()
        # # data augmentation
        # u_ori_1, i_ori_1 =F.dropout(u_ori, self.dropout), F.dropout(i_ori, self.dropout)
        # u_ori_2, i_ori_2 =F.dropout(u_ori, self.dropout), F.dropout(i_ori, self.dropout)
        # assert u_ori_1!=u_ori_2 and i_ori_1!=i_ori_2
        
        
        # # online network
        # u_online_ori, i_online_ori = self.forward()
        # t_feat_online, v_feat_online = None, None
        # if self.t_feat is not None:
        #     t_feat_online = self.text_trs(self.text_embedding.weight)
        # if self.v_feat is not None:
        #     v_feat_online = self.image_trs(self.image_embedding.weight)

        # with torch.no_grad():
        #     u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
        #     u_target.detach()
        #     i_target.detach()
        #     u_target = F.dropout(u_target, self.dropout)
        #     i_target = F.dropout(i_target, self.dropout)

        #     if self.t_feat is not None:
        #         t_feat_target = t_feat_online.clone()
        #         t_feat_target = F.dropout(t_feat_target, self.dropout)

        #     if self.v_feat is not None:
        #         v_feat_target = v_feat_online.clone()
        #         v_feat_target = F.dropout(v_feat_target, self.dropout)

        # u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        # users, items = interactions[0], interactions[1]
        # u_online = u_online[users, :]
        # i_online = i_online[items, :]
        # u_target = u_target[users, :]
        # i_target = i_target[items, :]

        # loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        # if self.t_feat is not None:
        #     t_feat_online = self.predictor(t_feat_online)
        #     t_feat_online = t_feat_online[items, :]
        #     t_feat_target = t_feat_target[items, :]
        #     loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
        #     loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        # if self.v_feat is not None:
        #     v_feat_online = self.predictor(v_feat_online)
        #     v_feat_online = v_feat_online[items, :]
        #     v_feat_target = v_feat_target[items, :]
            # loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
        #     loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        # loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        # loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        # return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
        #        self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.user_predictor(u_online), self.item_predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

