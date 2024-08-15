import os
# import wandb
import pickle
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.datasets import CitationFull
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.trainer.base import Trainer
# from framework.utils import negative_sampling_kg
import json
from scipy.sparse import dok_matrix
import numpy as np
from framework.utils import get_ii_constraint_mat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    # args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
            dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    popularity_bias = torch.load(os.path.join(args.data_dir, args.dataset, f'{args.random_seed}_popularity_bias.pt'))
    # wandb.init(project="GNNDelete",config=args)

    # print('Undirected dataset:', data)
    if 'x' in data.keys():   
        args.num_users, args.dim_users  = data['user'].x.size()
        args.num_items, args.dim_items  = data['item'].x.size()
    else:
        args.num_users = data.num_users
        args.num_items = data.num_items

    if args.gnn == 'UltraGCN':
        train_mat = dok_matrix((data.num_users, data.num_items), dtype=np.float32)
        for x in data.train_pos_edge_index.T:   
                train_mat[x[0], x[1]-data.num_users] = 1.0    
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, num_neighbors =args.num_neighbors) 
        ii_neighbor_mat, ii_constraint_mat =ii_neighbor_mat.to(device), ii_constraint_mat.to(device)

        items_D = np.sum(train_mat, axis = 0).reshape(-1)
        users_D = np.sum(train_mat, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1).to(device),
                        "beta_iD": torch.from_numpy(beta_iD).reshape(-1).to(device)}
    
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        data = data.to(device)
        model = get_model(args,  constraint_mat=constraint_mat, ii_constraint_mat=ii_constraint_mat, ii_neighbor_mat=ii_neighbor_mat).to(device)
    
    else:    
        train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        data.train_pos_edge_index = train_pos_edge_index
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        assert is_undirected(data.train_pos_edge_index)
        model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
        data = data.to(device)
        # wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = get_trainer(args)
    trainer.train(args, model, data, optimizer)

    # Test
    test_results = trainer.test(args, model, data, popularity_bias)
    trainer.save_log()
    result = test_results[-1]
    with open('output.txt', 'a') as f:
        f.write('\n')
        f.write(f'{args.dataset}, origianl, {args.gnn}, {args.random_seed}, {args.batch_size}, {args.epochs},  {args.df}, {args.df_size}, ')
        # f.write(f"{trainer.trainer_log['best_training_time']}, ")
        # f.write(f'{test_results[-1]}')  #文件的写操作
        # f.write(json.dumps(test_results[-1]))
        for i in test_results[:5]:
            f.write(f'{i}, ')
        # for i in  result["test_group_precision"].tolist():
        #     f.write(f'{round(i,4)}, ')
        # for i in  result["test_group_recall"].tolist():
        #     f.write(f'{round(i,4)}, ')  
        # for i in  result["test_group_nDCG"].tolist():
        #     f.write(f'{round(i,4)}, ')        

        print(test_results[:9])
        print(test_results[-1])


if __name__ == "__main__":
    main()
