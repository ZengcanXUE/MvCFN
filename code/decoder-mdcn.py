import torch
import torch.nn as nn
import utils
import numpy as np
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.in_channels = 1
        self.out_1 = 60
        self.out_2 = 80
        self.out_3 = 60

        self.bn0 = torch.nn.BatchNorm2d(1)  # in_channels
        self.bn1 = torch.nn.BatchNorm2d(self.out_1 + self.out_2 + self.out_3)
        # WN18RR and FB15k-237 para is different, fb15k-237: 200
        self.bn2 = torch.nn.BatchNorm1d(h_dim)  # embedding dimensions, fb15k-237: 450
        self.bn1_1 = torch.nn.BatchNorm2d(self.out_1)
        self.bn1_2 = torch.nn.BatchNorm2d(self.out_2)
        self.bn1_3 = torch.nn.BatchNorm2d(self.out_3)

        self.conv_drop = torch.nn.Dropout(cfg.conv_drop)  # fb15k-237:0.3
        self.fc_drop = torch.nn.Dropout(cfg.fc_drop)  # fb15k-237:0.5

        self.input_drop = torch.nn.Dropout(0.5)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.5)

        self.k_h = cfg.k_h  # fb15k-237:15
        self.k_w = cfg.k_w  # fb15k-237:30
        assert self.k_h * self.k_w == h_dim  # 15*30=450
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, stride=1, padding=0,
                                    kernel_size=ker_sz, bias=False)  # 2d conv
        # flat_sz_h = int(2 * self.k_h) - ker_sz + 1
        # flat_sz_w = self.k_w - ker_sz + 1
        flat_sz_h = 2 * self.k_h
        flat_sz_w = self.k_w
        self.flat_sz = flat_sz_h * flat_sz_w * (self.out_1+self.out_2+self.out_3)  # fc�ĳߴ�
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False)
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)  # fb15k-237:0.3

        fc1_length = self.in_channels * self.out_1 * 1 * 5
        self.fc1 = torch.nn.Linear(h_dim, fc1_length)
        fc2_length = self.in_channels * self.out_2 * 3 * 3
        self.fc2 = torch.nn.Linear(h_dim, fc2_length)
        fc3_length = self.in_channels * self.out_3 * 1 * 9
        self.fc3 = torch.nn.Linear(h_dim, fc3_length)

        # filter1_dim = self.in_channels * self.out_1 * 1 * 5
        # self.filter1 = torch.nn.Embedding(data.relations_num, filter1_dim, padding_idx=0)
        # filter2_dim = self.in_channels * self.out_2 * 3 * 3
        # self.filter3 = torch.nn.Embedding(data.relations_num, filter2_dim, padding_idx=0)
        # filter3_dim = self.in_channels * self.out_3 * 1 * 9
        # self.filter5 = torch.nn.Embedding(data.relations_num, filter3_dim, padding_idx=0)

        self.perm = 1
        self.embed_dim = h_dim
        self.k_hh = cfg.k_w
        self.k_ww = cfg.k_h
        self.chequer_perm = self.get_chequer_perm()

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        rel_perm = np.int32([np.random.permutation(self.embed_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_hh):
                for j in range(self.k_ww):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embed_dim)
                            rel_idx += 1
            comb_idx.append(temp)
        chequer_perm = torch.LongTensor(np.int32(comb_idx))
        return chequer_perm

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D
        entity = head.view(-1, 1, head.shape[-1])
        relation = rel.view(-1, 1, rel.shape[-1])
        relation_fc = rel

        f1 = self.fc1(relation_fc)
        f1 = f1.view(-1, self.in_channels, self.out_1, 1, 5)
        f1 = f1.view(entity.size(0) * self.in_channels * self.out_1, 1, 1, 5)

        f3 = self.fc2(relation_fc)
        f3 = f3.view(-1, self.in_channels, self.out_2, 3, 3)
        f3 = f3.view(entity.size(0) * self.in_channels * self.out_2, 1, 3, 3)

        f5 = self.fc3(relation_fc)
        f5 = f5.view(-1, self.in_channels, self.out_3, 1, 9)
        f5 = f5.view(entity.size(0) * self.in_channels * self.out_3, 1, 1, 9)

        # f1 = self.filter1(relation_id)
        # f1 = f1.reshape(entity.size(0) * self.in_channels * self.out_1, 1, 1, 5)
        # f3 = self.filter3(relation_id)
        # f3 = f3.reshape(entity.size(0) * self.in_channels * self.out_2, 1, 3, 3)
        # f5 = self.filter5(relation_id)
        # f5 = f5.reshape(entity.size(0) * self.in_channels * self.out_3, 1, 1, 9)

        # (b, 2, 200) → (b, 200, 2) → (b, 1, 20, 20)
        x = torch.cat([entity, relation], 1).transpose(1, 2).reshape(-1, 1, 2 * self.k_h, self.k_w)

        # sub_emb = head
        # rel_emb = rel
        # comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        # chequer_perm = comb_emb[:, self.chequer_perm]
        # x = chequer_perm.reshape(-1, 1, 2 * self.k_h, self.k_w)

        x = self.bn0(x)
        # x = self.input_drop(x)
        # (1 ,b, 20, 20)
        x = x.permute(1, 0, 2, 3)

        # (1, b*in*out, H-kH+1, W-kW+1)
        x1 = F.conv2d(x, f1, groups=entity.size(0), padding=(0, 2))
        x1 = x1.reshape(entity.size(0), self.out_1, 2 * self.k_h, self.k_w)
        x1 = self.bn1_1(x1)

        x3 = F.conv2d(x, f3, groups=entity.size(0), padding=(1, 1))
        x3 = x3.reshape(entity.size(0), self.out_2, 2 * self.k_h, self.k_w)
        x3 = self.bn1_2(x3)

        x5 = F.conv2d(x, f5, groups=entity.size(0), padding=(0, 4))
        x5 = x5.reshape(entity.size(0), self.out_3, 2 * self.k_h, self.k_w)
        x5 = self.bn1_3(x5)

        x = torch.cat([x1, x3, x5], dim=1)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)
        x = x.view(entity.size(0), -1)

        # (b, ent_dim)
        x = self.fc(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.hidden_drop(x)

        # inference
        # all_ent: (n_ent, h_dim),fb15k-237's h_dim is 450
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        x = torch.sigmoid(x)
        return x

        # # (batch, ent_dim)*(ent_dim, ent_num)=(batch, ent_num)
        # x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        # x += self.bias.expand_as(x)
        # pred = torch.sigmoid(x)
        #
        # return pred
