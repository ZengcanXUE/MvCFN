#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import utils
from utils import get_param
from decoder import ConvE


class MvCFN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']

        # entity embedding
        self.ent_emb = get_param(self.n_ent, h_dim)

        # gnn layer
        self.kg_n_layer = self.cfg.kg_layer
        # relation SE layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # entity SE layer
        self.node_layers = nn.ModuleList([NodeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # triple-rotate
        self.comp_layers2 = nn.ModuleList([CompLayer2(h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        # loss
        self.bce = nn.BCELoss()

        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.act = nn.Tanh()

    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding
        ent_emb, rel_emb = self.aggragate_emb(kg)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)

        return score

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        rel_emb_list = []

        for edge_layer, node_layer, comp_layer2, rel_emb in \
                zip(self.edge_layers, self.node_layers, self.comp_layers2, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb2 = comp_layer2(kg, ent_emb, rel_emb)
                    
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb2

            rel_emb_list.append(rel_emb)

        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb



class CompLayer2(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        
        self.granularity = getattr(self.cfg, 'granularity', 'rotation')  # rotation, complex, quaternion

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            
            if self.granularity == 'quaternion':
                
                node_chunks = kg.ndata['emb'].chunk(4, dim=-1)
                kg.ndata['node_a'] = node_chunks[0]  
                kg.ndata['node_b'] = node_chunks[1]  # i 
                kg.ndata['node_c'] = node_chunks[2]  # j 
                kg.ndata['node_d'] = node_chunks[3]  # k 
                
                edge_chunks = kg.edata['emb'].chunk(4, dim=-1)
                kg.edata['edge_a'] = edge_chunks[0]
                kg.edata['edge_b'] = edge_chunks[1]
                kg.edata['edge_c'] = edge_chunks[2]
                kg.edata['edge_d'] = edge_chunks[3]
                
                src_nodes = kg.edges()[0]
                
                h_a = kg.ndata['node_a'][src_nodes]
                h_b = kg.ndata['node_b'][src_nodes]
                h_c = kg.ndata['node_c'][src_nodes]
                h_d = kg.ndata['node_d'][src_nodes]
                
                r_a = kg.edata['edge_a']
                r_b = kg.edata['edge_b']
                r_c = kg.edata['edge_c']
                r_d = kg.edata['edge_d']
                
                #  q_result = q_rel * q_head
                t_a = r_a * h_a - r_b * h_b - r_c * h_c - r_d * h_d
                t_b = r_a * h_b + r_b * h_a + r_c * h_d - r_d * h_c
                t_c = r_a * h_c - r_b * h_d + r_c * h_a + r_d * h_b
                t_d = r_a * h_d + r_b * h_c - r_c * h_b + r_d * h_a
                
                kg.edata['comp_emb'] = torch.cat([t_a, t_b, t_c, t_d], dim=-1)
                
            elif self.granularity == 'complex':
                node_chunks = kg.ndata['emb'].chunk(2, dim=-1)
                kg.ndata['node_real'] = node_chunks[0]
                kg.ndata['node_imag'] = node_chunks[1]
                
                edge_chunks = kg.edata['emb'].chunk(2, dim=-1)
                kg.edata['edge_real'] = edge_chunks[0]
                kg.edata['edge_imag'] = edge_chunks[1]
                
                kg.apply_edges(fn.u_mul_e('node_real', 'edge_real', 'm1'))
                kg.apply_edges(fn.u_mul_e('node_imag', 'edge_imag', 'm2'))
                kg.edata['m_re'] = kg.edata['m1'] - kg.edata['m2']
                
                kg.apply_edges(fn.u_mul_e('node_real', 'edge_imag', 'm3'))
                kg.apply_edges(fn.u_mul_e('node_imag', 'edge_real', 'm4'))
                kg.edata['m_im'] = kg.edata['m3'] + kg.edata['m4']
                
                kg.edata['comp_emb'] = torch.cat([kg.edata['m_re'], kg.edata['m_im']], dim=-1)
                
            else:        
                node_chunks = kg.ndata['emb'].chunk(2, dim=-1)
                kg.ndata['node_real'] = node_chunks[0]
                kg.ndata['node_imag'] = node_chunks[1]
                
                edge_chunks = kg.edata['emb'].chunk(2, dim=-1)
        
                kg.edata['edge_real'] = torch.cos(edge_chunks[0])
                kg.edata['edge_imag'] = torch.sin(edge_chunks[1])
                
                kg.apply_edges(fn.u_mul_e('node_real', 'edge_real', 'm1'))
                kg.apply_edges(fn.u_mul_e('node_imag', 'edge_imag', 'm2'))
                kg.edata['m_re'] = kg.edata['m1'] - kg.edata['m2']
                
                kg.apply_edges(fn.u_mul_e('node_real', 'edge_imag', 'm3'))
                kg.apply_edges(fn.u_mul_e('node_imag', 'edge_real', 'm4'))
                kg.edata['m_im'] = kg.edata['m3'] + kg.edata['m4']
                
                kg.edata['comp_emb'] = torch.cat([kg.edata['m_re'], kg.edata['m_im']], dim=-1)

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # top-k sample
            kg.edata['weight'] = kg.edata['norm']
            kg = kg.to('cpu')
            sample_kg = dgl.sampling.select_topk(kg, 45, 'weight', edge_dir='in')
            sample_kg = sample_kg.to(self.device)

            # agg
            sample_kg.edata['comp_emb'] = sample_kg.edata['comp_emb'] * sample_kg.edata['norm']
            sample_kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = sample_kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # attention
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # attention
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb