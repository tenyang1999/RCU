import os
import time
# import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics.pairwise import cosine_similarity
from .base import Trainer
# from 
# from ..evaluation import *
from ..utils import *
from torch.autograd import grad
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CEUTrainer(Trainer):

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None)
    
    def unlearn(self, args, model, data, data_prime, df_edge, device, return_bound=False):
        _model = copy.deepcopy(model)
        # parameters = [p for p in _model.parameters() if p.requires_grad]
        infected_nodes = data.infected_nodes(df_edge, len(args.hidden) + 1)
        infected_indices = np.where(np.in1d(np.array(data.train_set.nodes), np.array(infected_nodes)))[0]
        infected_nodes = torch.tensor(infected_nodes, device=device)
        infected_labels = torch.tensor(data.labels_of_nodes(infected_nodes.cpu().tolist()), device=device)
        # print('Number of nodes:', len(infected_nodes))

        t0 = time.time()
        #args.appr == 'cg'
        if return_bound:
            infl, bound = influence(args, _model, data, data_prime, infected_nodes, infected_labels, device=device, use_torch=True, return_norm=return_bound)
        else:
            infl = influence(args, _model, data, data_prime, infected_nodes, infected_labels, device=device, use_torch=True, return_norm=return_bound)

        # print('!!!!!!', torch.linalg.norm(torch.cat([ii.view(-1) for ii in infl])))
        self._update_model_weight(_model, infl)
        
        duration = time.time() - t0
        if return_bound:
            return _model, duration, bound.item(), infl
        else:
            return _model, duration
        
    def _update_model_weight(self, model, influence):
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, influence)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])
            
    def influence(args, model, data, data_prime, infected_nodes, infected_labels, use_torch=True, device=torch.device('cpu'), return_norm=False):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if args.transductive_edge:
            edge_index = torch.tensor(data.edges, device=device).t()
            edge_index_prime = torch.tensor(data_prime.edges, device=device).t()
        else:
            edge_index = torch.tensor(data.train_edges, device=device).t()
            edge_index_prime = torch.tensor(data_prime.train_edges, device=device).t()

        p = 1 / (data.num_train_nodes)

        # t1 = time.time()
        model.eval()
        y_hat = model(infected_nodes, edge_index_prime)
        loss1 = model.loss_sum(y_hat, infected_labels)
        g1 = grad(loss1, parameters)
        # print(f'CEU, grad new duration: {(time.time() - t1):.4f}.')

        # t1 = time.time()
        y_hat = model(infected_nodes, edge_index)
        loss2 = model.loss_sum(y_hat, infected_labels)
        g2 = grad(loss2, parameters)
        # print(f'CEU, grad old duration: {(time.time() - t1):.4f}.')

        v = [gg1 - gg2 for gg1, gg2 in zip(g1, g2)]
        # ihvp = inverse_hvp(data, model, edge_index, v, args.damping, device)
        # ihvp, (cg_grad, status) = inverse_hvp_cg_sep(data, model, edge_index, v, args.damping, device, use_torch)

        # t1 = time.time()
        if args.model == 'sgc':
            ihvp, (cg_grad, status) = inverse_hvp_cg_sgc(data, model, edge_index, v, args.lam, device)
        else:
            if len(args.hidden) == 0:
                ihvp, (cg_grad, status) = inverse_hvp_cg(data, model, edge_index, v, args.damping, device, use_torch)
            else:
                ihvp, (cg_grad, status) = inverse_hvp_cg_sep(data, model, edge_index, v, args.damping, device, use_torch)
        # print(f'CEU, hessian inverse duration: {(time.time() - t1):.4f}.')
        
        I = [- p * i for i in ihvp]

        # print('-------------------------')
        # print('Norm:', [torch.norm(ii) for ii in I])
        # print('v norm:', [torch.norm(vv) for vv in v])
        # print('CG gradient:', cg_grad)
        # print('status:', status)
        # print('-------------------------')


        if return_norm:
            return I, (torch.norm(torch.cat([i.view(-1) for i in ihvp])) ** 2) * (p ** 2)
        else:
            return I
    # return inverse_hvps, loss, status

    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = 0

        if self.with_fea:
            dim_users, dim_items = data['user'].x.size(1), data['item'].x.size(1)
            users_lin = torch.rand(dim_users, args.out_dim).to(device)
            items_lin = torch.rand(dim_items, args.out_dim).to(device)
            user_emb = torch.mm(data['user'].x, users_lin)
            item_emb = torch.mm(data['item'].x, items_lin)
            X = torch.cat([user_emb, item_emb])
            x = torch.tensor(X, requires_grad= True)
        else:
            x = None
        # data = data.to(device)
        # dr_edge = data.train_pos_edge_index[:, data.dr_mask].to(device)
        df_edge = data.train_pos_edge_index[:, data.df_mask].to(device)
        early_stop_cnt = 0

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            # train_edge_index = data.train_pos_edge_index

            loader = torch.utils.data.DataLoader(
                range(df_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )
            epoch_loss = epoch_time = total_examples = 0
            for step, ind in enumerate(tqdm(loader, desc='Step',position=0, leave=True)):
                start = time.time()
                # Positive and negative sample

                batch_df_edge = df_edge[:, ind].to(device)

                # user_indices, pos_item_indices = pos_edge_index
                # neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
                # indices = [user_indices, pos_item_indices, neg_item_indices]
                
                # user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch.size(0), edge_index)

                # neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                # edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                optimizer.zero_grad()

                _data = copy.deepcopy(data)
                _data.remove_edges(batch_df_edge)

                unlearn_model, unlearn_time = unlearn(args, model, data, _data, batch_df_edge, device)
                
                # rank, emb = model(data['user'].x, data[item].x , train_edge_index, edge_label_index)
                # emb= model(x, train_edge_index)
                # rank = model.decode(emb, edge_label_index)
                
                # loss = self.loss_fn(rank, indices, emb)
                    
                loss.backward()
                optimizer.step()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                # wandb.log(log)
                # msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                # tqdm.write(' | '.join(msg))

                epoch_loss += float(loss.item()) * (rank.numel()/2)
                total_examples += (rank.numel()/2)
                epoch_time += time.time() - start

            train_loss = epoch_loss/ total_examples

            if (epoch+1) % args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, stage='val')
                # precision, recall, ndcg = self.test(args, model, data, k=20)
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                train_log = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'epoch_time': epoch_time
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if best_recall < recall:
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg

                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(emb, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
                    early_stop_cnt = 0
                else:
                    early_stop_cnt +=1

            if early_stop_cnt >= 10:
                print(f'early stop training at epoch {epoch}')
                break

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d},\
        best_precison = {best_precision}, best_recall = {best_recall}, best_ndcg = {best_ndcg} ')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_precison'] = best_precision
        self.trainer_log['best_recall'] = best_recall
        self.trainer_log['best_ndcg'] = best_ndcg
        
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])

        return train_edge_index

