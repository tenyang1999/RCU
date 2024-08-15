import os
import time
# import wandb
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from .base import Trainer
# from ..evaluation import *
from ..utils import *


class DtdTrainer(Trainer):
    '''This code is adapte from https://github.com/ChrisWaites/descent-to-delete'''
    
    def compute_sigma(self, num_examples, iterations, lipshitz, smooth, strong, epsilon, delta):
        """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""

        print('delta', delta)
        gamma = (smooth - strong) / (smooth + strong)
        numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
        denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
        # print('sigma', numerator, denominator, numerator / denominator)
    
        return numerator / denominator

    def publish(self, model, sigma):
        """Publishing function which adds Gaussian noise with scale sigma."""

        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(p + torch.empty_like(p).normal_(0, sigma))

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):

    # def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_valid_loss = 100000
        best_recall = best_precision = best_ndcg = 0

        # # MI Attack before unlearning
        # if attack_model_all is not None:
        #     mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
        #     self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
        #     self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        # if attack_model_sub is not None:
        #     mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
        #     self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
        #     self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before
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
        train_edge_index = data.train_pos_edge_index[:, data.dr_mask]
        early_stop_cnt = 0
        
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()


            pos_edge_index = train_edge_index.to(device)

            user_indices, pos_item_indices = pos_edge_index
            neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(user_indices.numel(), ), device=device)
            indices = [user_indices, pos_item_indices, neg_item_indices]
            
            # user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch.size(0), edge_index)

            neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
        
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            # Positive and negative sample
            # neg_edge_index = negative_sampling(
            #     edge_index=data.train_pos_edge_index[:, data.dr_mask],
            #     num_nodes=data.num_nodes,
            #     num_neg_samples=data.dr_mask.sum())
            # loader = torch.utils.data.DataLoader(
            #     range(train_edge_index.size(1)),
            #     shuffle=True,
            #     batch_size=args.batch_size,
            # )
            emb= model(x, train_edge_index)
            rank = model.decode(emb, edge_label_index)
                
            loss = self.loss_fn(rank, indices, emb)
            # z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            # logits = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            # label = get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            # loss = F.binary_cross_entropy_with_logits(logits, label)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
            }
            # wandb.log(log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))

            # valid_loss, auc, aup, df_logt, logit_all_pair = self.eval(model, data, 'val')
            precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, group= popularity_bias, stage='val', score=None)

            self.trainer_log['log'].append({
                # 'dt_loss': valid_loss,
                'dt_auc': df_auc,
                'dt_aup': df_aup
            })
        
        train_size = data.dr_mask.sum().cpu().item()
        sigma = self.compute_sigma(train_size, args.epochs, 1 + args.weight_decay, 4 - args.weight_decay, args.weight_decay, 5, 1 / train_size / train_size)
        
        self.publish(model, sigma)

        self.trainer_log['sigma'] = sigma
        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
