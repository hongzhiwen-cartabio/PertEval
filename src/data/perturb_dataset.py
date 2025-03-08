import torch
import os
from pathlib import PurePath
import anndata
import pickle
import gzip
import gdown
import warnings
import gc

import numpy as np
import scanpy as sc
import pickle as pkl
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from hydra.errors import HydraException
from scipy.stats import pearsonr
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from joblib import Parallel, delayed

from src.utils.spectra import get_splits
from src.data.components import embeddings
import json
import pandas as pd


class PerturbData(Dataset):
    ctrl_expr_cache = None

    def __init__(self, adata, data_path, spectral_parameter, spectra_params, fm, stage, **kwargs):
        self.data_name = PurePath(data_path).parts[-1]
        self.data_path = data_path
        self.spectral_parameter = spectral_parameter
        #self.spectra_params = spectra_params
        self.stage = stage
        self.fm = fm
        self.data_processor = None
        self.deg_dict = None
        self.basal_ctrl_adata = None
        self.genes = None
        self.all_perts_train = None
        self.all_perts_test = None

        if kwargs:
            if 'deg_dict' in kwargs and 'perturbation' in kwargs:
                self.deg_dict = kwargs['deg_dict']
                self.perturbation = kwargs['perturbation']
            else:
                raise HydraException("kwargs can only contain 'perturbation' and 'deg_dict' keys!")

        if self.fm == 'mean':
            # use raw_expression data to calculate mean expression
            self.fm = 'raw_expression'

        assert self.fm in ["raw_expression", "scgpt", "geneformer", "scfoundation", "scbert", "uce"], \
            "fm must be set to 'raw_expression', 'scgpt', 'geneformer', 'scfoundation', 'scbert', or 'uce'!"

        feature_path = f"{self.data_path}/input_features/{self.fm}"

        if not os.path.exists(feature_path):
            os.makedirs(feature_path, exist_ok=True)
        
        # Data loading logic

        data_file = f"{feature_path}/train_data.pkl.gz"
        if not os.path.exists(data_file):
            (self.X_train, self.train_target,
             self.X_val, self.val_target,
             self.X_test, self.test_target,
             self.ctrl_expr, _) = self.preprocess_and_featurise(adata)
        else:
            self._load_preprocessed_data(feature_path)

    def _load_preprocessed_data(self, feature_path):
        self.basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_pp_filtered.h5ad")
        with gzip.open(f"{feature_path}/train_data.pkl.gz", "rb") as f:
            self.X_train, self.train_target = pkl.load(f)
        with gzip.open(f"{feature_path}/test_data.pkl.gz", "rb") as f:
            self.X_test, self.test_target = pkl.load(f)
        with open(f"{self.data_path}/raw_expression_{self.data_name}_pp_filtered.pkl", "rb") as f:
            self.ctrl_expr = pkl.load(f)

    def preprocess_and_featurise(self, adata, fold=0):
        self.genes = adata.var.index.tolist()
        feature_path = f"{self.data_path}/input_features/{self.fm}"
        os.makedirs(feature_path, exist_ok=True)
    
    # 1. 加载预训练嵌入
        # ctrl_emb_path = f"{self.data_path}/embeddings/{self.data_name}_scgpt_fm_ctrl.pkl.gz"
        # pert_emb_path = f"{self.data_path}/embeddings/{self.data_name}_scgpt_fm_pert.pkl.gz"
        embs = anndata.read_h5ad(f'./data/{self.data_name}_{self.fm}.h5ad')
        ctrl_emb = embs.obsm['ctrl_emb']
        pert_emb = embs.obsm['pert_emb']
        # assert (ctrl_emb.obs_names == pert_emb.obs_names).all()

        # assert os.path.exists(ctrl_emb_path) and os.path.exists(pert_emb_path), \
        # "预生成嵌入文件不存在！请确保 scGPT 的 ctrl/pert 嵌入已生成并保存在指定路径"
        #
        # with gzip.open(ctrl_emb_path, "rb") as f:
        #     fm_ctrl_data = pkl.load(f)  # 格式假设为 (num_ctrl_cells, embedding_dim)
        # with gzip.open(pert_emb_path, "rb") as f:
        #     fm_pert_data = pkl.load(f)  # 格式假设为 {pert_name: np.array of embeddings}

    # 2. 获取扰动列表并划分训练测试集
        pert_list = embs.obs['guide_ids'].unique()
        with open(f'./data/{self.data_name}_fold_indices.json') as f:
            splits = json.load(f)
        pert_names = pd.read_csv(f'./data/{self.data_name}_fold_names.csv').values[:, 0]
        train_perts = pert_names[splits[self.spectral_parameter]['train_indices']]
        train_perts, valid_perts = train_test_split(pert_list, test_size=0.2, random_state=42)
        test_perts = pert_names[splits[self.spectral_parameter]['test_indices']]
        adata = adata[embs.obs_names].copy()
        adata.obs['condition'] = adata.obs['condition'].str.upper()

    
    # 4. 构造目标输出
        X_train = torch.FloatTensor(np.concatenate([ctrl_emb[embs.obs['guide_ids'].isin(train_perts)],
                                                        pert_emb[embs.obs['guide_ids'].isin(train_perts)]], 1))
        X_val = torch.FloatTensor(np.concatenate([ctrl_emb[embs.obs['guide_ids'].isin(valid_perts)],
                                                    pert_emb[embs.obs['guide_ids'].isin(valid_perts)]], 1))
        X_test= torch.FloatTensor(np.concatenate([ctrl_emb[embs.obs['guide_ids'].isin(test_perts)],
                                                       pert_emb[embs.obs['guide_ids'].isin(test_perts)]], 1))
        train_target = torch.FloatTensor(
            np.concatenate([adata[adata.obs['condition'] == pert].X.toarray() for pert in train_perts]))
        val_target = torch.FloatTensor(
            np.concatenate([adata[adata.obs['condition'] == pert].X.toarray() for pert in valid_perts]))
        test_target = torch.FloatTensor(
            np.concatenate([adata[adata.obs['condition'] == pert].X.toarray() for pert in test_perts]))
        assert train_target.shape[0] == X_train.shape[0] and X_test.shape[0] == test_target.shape[0], (str(train_target.shape[0])+','+str(X_train.shape[0]))
        # train_target = torch.FloatTensor([fm_pert_data[pert].mean(axis=0) for pert in train_perts])
        # test_target = torch.FloatTensor([fm_pert_data[pert].mean(axis=0) for pert in test_perts])
    
    # 5. 保存预处理数据
    #     with gzip.open(f"{feature_path}/train_data.pkl.gz", "wb") as f:
    #         pkl.dump((X_train, train_target), f)
    #     with gzip.open(f"{feature_path}/test_data.pkl.gz", "wb") as f:
    #         pkl.dump((X_test, test_target), f)
    
        return X_train, train_target, X_val, val_target, X_test, test_target, None, None


    def preprocess_and_featurise_replogle(self, adata):
    
        # pert_adata = adata[adata.obs['condition'] != 'ctrl', :]
        # all_perts = list(set(pert_adata.obs['condition'].to_list()))
        # train_perts, test_perts = train_test_split(all_perts, test_size=0.2, random_state=42)

        nonzero_genes = (adata.X.sum(axis=0) > 5)
        filtered_adata = adata[:, nonzero_genes]
        adata.obs['condition'] = adata.obs['perturbation'].replace('control', 'ctrl')

        self.genes = adata.var.index.to_list()
        genes_and_ctrl = self.genes + ['ctrl']
        ensembl_id = adata.var['ensembl_id']
        ensembl_ids = ensembl_id.apply(lambda x: x).tolist()

        adata = filtered_adata[adata.obs['condition'].isin(genes_and_ctrl), :]
                # 获取所有扰动列表并划分
        

        train, test, pert_list = get_splits.spectra(adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )

        print(f"Replogle dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :]
        pert_adata = adata[adata.obs['condition'] != 'ctrl', :]
        all_perts = list(set(pert_adata.obs['condition'].to_list()))

        num_cells = ctrl_adata.shape[0]
        num_perts = len(all_perts)

        mask = np.zeros((num_cells, num_perts), dtype=bool)

        for idx, pert in enumerate(all_perts):
            mask = self.sg_pert_mask(mask, pert, idx, ctrl_adata)

        mask_df = pd.DataFrame(mask, columns=all_perts)
        mask_df.to_pickle(f"{self.data_path}/replogle_mask_df.pkl")

        mask_df_cells = mask_df.any(axis=0)
        unique_perts = list(mask_df.columns[mask_df_cells])

        gene_to_ensg = dict(zip(self.genes, ensembl_ids))

        basal_ctrl_path = f"{self.data_path}/basal_ctrl_replogle_pp_filtered.h5ad"

        if not os.path.exists(basal_ctrl_path):
            ctrl_X = ctrl_adata.X.toarray()
            basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
            subset_size = 500

            for cell in tqdm(range(pert_adata.shape[0])):
                subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                basal_ctrl_X[cell, :] = subset.mean(axis=0)

            basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)
            basal_ctrl_adata.write(basal_ctrl_path, compression='gzip')
        else:
            basal_ctrl_adata = sc.read_h5ad(basal_ctrl_path)

        if not os.path.exists(f"{self.data_path}/raw_expression_replogle_pp_filtered.pkl"):
            ctrl_expr = basal_ctrl_adata[basal_ctrl_adata.obs['condition'].isin(unique_perts), :]
            ctrl_expr = ctrl_expr.X.toarray()
            with open(f"{self.data_path}/raw_expression_replogle_pp_filtered.pkl", "wb") as f:
                pkl.dump(ctrl_expr, f)
        else:
            with open(f"{self.data_path}/raw_expression_replogle_pp_filtered.pkl", "rb") as f:
                ctrl_expr = pkl.load(f)

        with open(f"{self.data_path}/raw_expression_replogle_pp_filtered.pkl", "rb") as f:
            ctrl_expr = pkl.load(f)
        basal_ctrl_adata = sc.read_h5ad(basal_ctrl_path)
        pert_adata = sc.read_h5ad(f"{self.data_path}/replogle_pp_pert_filtered.h5ad")

        ctrl_cell_conditions = basal_ctrl_adata.obs['condition'].to_list()
        pert_cell_conditions = pert_adata.obs['condition'].to_list()

        assert ctrl_cell_conditions == pert_cell_conditions, ("Watch out! Cell conditions in control and perturbation "
                                                              "datasets are not the same, or are not indexed the same!")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
        test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

        self.all_perts_train = train_target.obs['condition'].values
        self.all_perts_test = test_target.obs['condition'].values

        if not os.path.exists(f"{self.data_path}/target_perts"):
            os.makedirs(f"{self.data_path}/target_perts")

        with open(f"{self.data_path}/target_perts/all_perts_test_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump(self.all_perts_test, f)

        if not os.path.exists(f"{self.data_path}/pert_corrs.pkl"):
            all_gene_expression = basal_ctrl_adata.X

            processed_perts = []
            pert_corrs = {}
            for pert in tqdm(unique_perts, total=len(unique_perts)):
                correlations = np.zeros(basal_ctrl_adata.shape[1])
                if pert in processed_perts:
                    continue
                ensg_id = gene_to_ensg[pert]
                pert_idx = basal_ctrl_adata.var_names.get_loc(ensg_id)
                basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
                for i in range(all_gene_expression.shape[1]):
                    corr = np.corrcoef(basal_expr_pert, all_gene_expression[:, i])[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    correlations[i] = corr
                processed_perts.append(pert)
                pert_corrs[pert] = correlations

            with open(f"{self.data_path}/pert_corrs.pkl", "wb") as f:
                pkl.dump(pert_corrs, f)
        else:
            with open(f"{self.data_path}/pert_corrs.pkl", "rb") as f:
                pert_corrs = pkl.load(f)

        num_ctrl_cells = basal_ctrl_adata.shape[0]
        num_train_cells = train_target.shape[0]
        num_test_cells = test_target.shape[0]
        num_genes = basal_ctrl_adata.shape[1]

        random_train_mask = np.random.randint(0, num_ctrl_cells, num_train_cells)
        random_test_mask = np.random.randint(0, num_ctrl_cells, num_test_cells)

        pert_corr_train = np.zeros((num_train_cells, num_genes))
        for i, pert in tqdm(enumerate(self.all_perts_train), total=len(self.all_perts_train)):
            pert_corr_train[i, :] = pert_corrs[pert]

        pert_corr_test = np.zeros((num_test_cells, num_genes))
        for i, pert in tqdm(enumerate(self.all_perts_test), total=len(self.all_perts_test)):
            pert_corr_test[i, :] = pert_corrs[pert]

        train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
        test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

        raw_X_train = np.concatenate((train_input_expr, pert_corr_train), axis=1)
        X_test = np.concatenate((test_input_expr, pert_corr_test), axis=1)

        raw_train_target = train_target.X.toarray()

        X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                      raw_train_target,
                                                                      test_size=0.2)

        X_train = torch.from_numpy(X_train)
        train_target = torch.from_numpy(train_targets)
        X_val = torch.from_numpy(X_val)
        val_target = torch.from_numpy(val_targets)
        X_test = torch.from_numpy(X_test)
        test_target = torch.from_numpy(test_target.X.toarray())

        save_path = f"{self.data_path}/input_features/{self.fm}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with gzip.open(f"{self.data_path}/input_features/{self.fm}/train_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_train, train_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/val_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_val, val_target), f)
        with gzip.open(f"{self.data_path}/input_features/{self.fm}/test_data_{self.spectral_parameter}.pkl.gz",
                       "wb") as f:
            pkl.dump((X_test, test_target), f)

        del basal_ctrl_adata, control_genes, pert_genes, pert_cell_conditions, ctrl_cell_conditions
        del train_perts, test_perts, train_target, all_perts_train
        del pert_corrs
        del random_train_mask, train_input_expr, raw_X_train, raw_train_target
        del X_train, X_val, train_targets, val_targets, X_test

        raise HydraException(f"Completed preprocessing and featurisation of split {self.spectral_parameter}. Moving "
                             f"on the next multirun...")

        # return X_train, train_target, X_val, val_target, X_test, test_target, ctrl_expr, self.all_perts_test
    
    def preprocess_replogle(self, adata):
         adata.obs['condition'] = adata.obs['perturbation'].replace('control', 'ctrl')
       # 基因过滤
         self.genes = adata.var.index.to_list()
         genes_and_ctrl = self.genes + ['ctrl']
         adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]
         #genes = adata.var.index.to_list()
    #     genes_and_ctrl = genes + ['ctrl']

    #     # we remove the cells with perts that are not in the genes because we need gene expression values
    #     # to generate an in-silico perturbation embedding
    #     adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]
         unique_perts = list(set(adata.obs['condition'].to_list()))
         unique_perts.remove('ctrl')
         # 划分训练测试集 (8:2)
             # 划分训练测试集 (8:2)
         train_perts, test_perts = train_test_split(
            unique_perts, 
            test_size=0.2,
            random_state=42,
            shuffle=True
         )
         pert_list = unique_perts  # 保持原有变量结构

    #      train, test, pert_list = get_splits.spectra(adata,
    # #                                                 self.data_path,
    # #                                                 self.spectra_params,
    # #                                                 self.spectral_parameter
    # #                                                 )

         sc.pp.normalize_total(adata)
         sc.pp.log1p(adata)
         sc.pp.highly_variable_genes(adata, n_top_genes=2000)
         hvg_genes = adata.var_names[adata.var['highly_variable']]
    #     highly_variable_genes = list(adata.var_names[adata.var['highly_variable']])
         missing_perts = list(set(unique_perts) - set(hvg_genes))
         combined_genes = list(set(hvg_genes) | set(missing_perts))

         adata = adata[:, combined_genes]
         #missing_perts = list(set(unique_perts) - set(highly_variable_genes))
    #     combined_genes = list(set(highly_variable_genes) | set(missing_perts))
    #     adata = adata[:, combined_genes]

         ctrl_adata = adata[adata.obs['condition'] == 'ctrl', :]
         pert_adata = adata[adata.obs['condition'] != 'ctrl', :]

         num_cells = ctrl_adata.shape[0]
         num_perts = len(pert_list)
         mask_path = f"{self.data_path}/{self.data_name}_mask_df.pkl"
    #     mask = np.zeros((num_cells, num_perts), dtype=bool)

         if not os.path.exists(f"{self.data_path}/{self.data_name}_mask_df.pkl"):
             mask = np.zeros((num_cells, num_perts), dtype=bool)

             for idx, pert in tqdm(enumerate(pert_list), total=len(pert_list)):
                 try:
                     pert_idx = adata.var_names.get_loc(pert)
                     non_zero_indices = ctrl_adata[:, pert_idx].X.nonzero()[0]
                
                     if len(non_zero_indices) == 0:
                         print(f"Warning: {pert} has no non-zero cells")
                         continue
                    
                     sample_num = min(500, len(non_zero_indices))
                     sampled_indices = np.random.choice(
                         non_zero_indices, 
                         size=sample_num, 
                         replace=False
                     )
                     mask[sampled_indices, idx] = True
                 except KeyError:
                     print(f"Skipping {pert} not found in var_names")
                     continue
    #             pert_idx = combined_genes.index(pert)
    #             non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
    #             num_non_zeroes = len(non_zero_indices)

    #             if len(non_zero_indices) < 500:
    #                 sample_num = num_non_zeroes
    #             else:
    #                 sample_num = 500

    #             sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)

    #             mask[sampled_indices, i] = True

             mask_df = pd.DataFrame(mask, columns=pert_list)

             mask_df.to_pickle(f"{self.data_path}/{self.data_name}_mask_df.pkl")
         else: 
             mask_df = pd.read_pickle(mask_path)
    #     if not os.path.exists(f"{self.data_path}/all_perts.pkl"):
    #         with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
    #             pkl.dump(unique_perts, f)

    #     return ctrl_adata, pert_adata, train, test, pert_list
         basal_ctrl_path = f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"
         if not os.path.exists(basal_ctrl_path):
             ctrl_X = ctrl_adata.X.toarray()
             basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
             subset_size = 500
             
             # 并行化采样
             def process_cell(cell_idx):
                 return ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :].mean(axis=0)
             results = Parallel(n_jobs=8)(
                 delayed(process_cell)(i) 
                 for i in tqdm(range(pert_adata.shape[0]))
             )
        
             for i, result in enumerate(results):
                 basal_ctrl_X[i, :] = result

             basal_ctrl_adata = anndata.AnnData(
                 X=basal_ctrl_X,
                 obs=pert_adata.obs.copy(),
                 var=ctrl_adata.var.copy()
              )
        #               results = Parallel(n_jobs=8)(
        #     delayed(process_cell)(i) 
        #     for i in tqdm(range(pert_adata.shape[0]))
        
        # for i, result in enumerate(results):
        #     basal_ctrl_X[i, :] = result

        # basal_ctrl_adata = anndata.AnnData(
        #     X=basal_ctrl_X,
        #     obs=pert_adata.obs.copy(),
        #     var=ctrl_adata.var.copy()
        # )
        #basal_ctrl_adata = sc.read_h5ad(basal_ctrl_path)
             basal_ctrl_adata.write(basal_ctrl_path, compression='gzip')
         else:
             basal_ctrl_adata = sc.read_h5ad(basal_ctrl_path)
            # 保存必要元数据
         meta_path = f"{self.data_path}/preprocess_meta.pkl"
         if not os.path.exists(meta_path):
             meta_data = {
                 'train_perts': train_perts,
                 'test_perts': test_perts,
                 'pert_list': pert_list,
                 'gene_list': combined_genes
             }
             with open(meta_path, 'wb') as f:
                 pkl.dump(meta_data, f)

    # 内存清理
         del ctrl_X, basal_ctrl_X
         gc.collect()

         return ctrl_adata, pert_adata, train_perts, test_perts, pert_list



    def featurise_replogle(self, pert_adata, pert_list, ctrl_adata, train, test):
        print(f"{self.data_name} dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
            ctrl_X = ctrl_adata.X.toarray()
            basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
            subset_size = 500

            for cell in tqdm(range(pert_adata.shape[0])):
                subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                basal_ctrl_X[cell, :] = subset.mean(axis=0)

            # we add pert_adata to obs because we want to pair control expression to perturbed cells
            basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

            # noinspection PyTypeChecker
            basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")
        else:
            basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")

        # these just need to be the same between datasets, irrespective of order
        control_genes = sorted(basal_ctrl_adata.var_names.to_list())
        pert_genes = sorted(pert_adata.var_names.to_list())

        # these need to be paired between datasets, and in the same order
        pert_cell_conditions = pert_adata.obs['condition'].to_list()
        ctrl_cell_conditions = basal_ctrl_adata.obs['condition'].to_list()

        assert control_genes == pert_genes, "Watch out! Genes in control and perturbation datasets do not match!"

        assert ctrl_cell_conditions == pert_cell_conditions, ("Watch out! Cell conditions in control and perturbation "
                                                              "datasets are not the or same, or are not indexed the "
                                                              "same!")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        train_target = pert_adata[pert_adata.obs['condition'].isin(train_perts), :]
        test_target = pert_adata[pert_adata.obs['condition'].isin(test_perts), :]

        all_perts_train = train_target.obs['condition'].values
        all_perts_test = test_target.obs['condition'].values

        if not os.path.exists(f"{self.data_path}/pert_corrs.pkl.gz"):
            all_gene_expression = basal_ctrl_adata.X

            results = []

            basal_ctrl_adata.X = basal_ctrl_adata.X.astype(np.float32)
            all_gene_expression = all_gene_expression.astype(np.float32)

            for pert in tqdm(pert_list, total=len(pert_list)):
                pert, correlations = self.compute_pert_correlation(pert, basal_ctrl_adata, all_gene_expression)
                results.append((pert, correlations))
            pert_corrs = {pert: corr for pert, corr in results}

            with gzip.open(f"{self.data_path}/pert_corrs.pkl.gz", "wb") as f:
                pkl.dump(pert_corrs, f)
        else:
            with gzip.open(f"{self.data_path}/pert_corrs.pkl.gz", "rb") as f:
                pert_corrs = pkl.load(f)

        num_ctrl_cells = basal_ctrl_adata.shape[0]
        num_train_cells = train_target.shape[0]
        num_test_cells = test_target.shape[0]
        num_genes = basal_ctrl_adata.shape[1]

        pert_corr_train = np.zeros((num_train_cells, num_genes))
        for i, pert in tqdm(enumerate(all_perts_train), total=len(all_perts_train)):
            pert_corr_train[i, :] = pert_corrs[pert]

        pert_corr_test = np.zeros((num_test_cells, num_genes))
        for i, pert in tqdm(enumerate(all_perts_test), total=len(all_perts_test)):
            pert_corr_test[i, :] = pert_corrs[pert]

        print("\n\nPertubation correlation features computed.\n\n")

        random_train_mask = np.random.randint(0, num_ctrl_cells, num_train_cells)
        random_test_mask = np.random.randint(0, num_ctrl_cells, num_test_cells)

        train_input_expr = basal_ctrl_adata[random_train_mask, :].X.toarray()
        test_input_expr = basal_ctrl_adata[random_test_mask, :].X.toarray()

        print("\n\nInput expression data generated.\n\n")

        raw_X_train = np.concatenate((train_input_expr, pert_corr_train), axis=1)
        raw_train_target = train_target.X.toarray()

        X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                      raw_train_target,
                                                                      test_size=0.2)

        X_train = torch.from_numpy(X_train)
        train_target = torch.from_numpy(train_targets)
        X_val = torch.from_numpy(X_val)
        val_target = torch.from_numpy(val_targets)
        X_test = torch.from_numpy(np.concatenate((test_input_expr, pert_corr_test), axis=1))
        test_target = torch.from_numpy(test_target.X.toarray())

        # save data as pickle without gzip
        with open(f"{self.data_path}/input_features/{self.fm}/train_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_train, train_target), f)
        with open(f"{self.data_path}/input_features/{self.fm}/val_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_val, val_target), f)
        with open(f"{self.data_path}/input_features/{self.fm}/test_data_{self.spectral_parameter}.pkl", "wb") as f:
            pkl.dump((X_test, test_target), f)

        # del basal_ctrl_adata, control_genes, pert_genes, pert_cell_conditions, ctrl_cell_conditions
        # del train_perts, test_perts, train_target, all_perts_train
        # del pert_corrs
        # del random_train_mask, train_input_expr, raw_X_train, raw_train_target
        # del X_train, X_val, train_targets, val_targets, X_test
        #
        # raise HydraException(f"Completed preprocessing and featurisation of split {self.spectral_parameter}. Moving "
        #                      f"on the next multirun...")

        return X_train, train_target, X_val, val_target, X_test, test_target

    @staticmethod
    def compute_pert_correlation(basal_expr_pert, all_gene_expression):
        basal_mean = basal_expr_pert.mean()
        basal_centered = basal_expr_pert - basal_mean
        all_gene_mean = all_gene_expression.mean(axis=0)
        all_gene_centered = all_gene_expression - all_gene_mean

        numerator = np.dot(basal_centered, all_gene_centered)
        denominator = np.linalg.norm(basal_centered) * np.linalg.norm(all_gene_centered, axis=0)
        correlations = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        return correlations

    def sg_pert_mask(self, mask, pert, idx, ctrl_adata):
        pert_idx = self.genes.index(pert)
        non_zero_indices = ctrl_adata[:, pert_idx].X.sum(axis=1).nonzero()[0]
        num_non_zeroes = len(non_zero_indices)

        if len(non_zero_indices) == 0:
            print(f"{pert} has no nonzero values in the control dataset! Kicking it from the analysis.")
            return mask
        elif len(non_zero_indices) < 500:
            sample_num = num_non_zeroes
        else:
            sample_num = 500

        sampled_indices = np.random.choice(non_zero_indices, sample_num, replace=False)
        mask[sampled_indices, idx] = True

        return mask

    def __getitem__(self, index):
        if self.stage == "train":
            return self.X_train[index], self.train_target[index], self.ctrl_expr[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index], self.ctrl_expr[index]
        elif self.stage == "test" and self.deg_dict is None:
            if self.all_perts_test is not None:
                return self.X_test[index], self.test_target[index], self.all_perts_test[index], self.ctrl_expr[index]
            else:
                return self.X_test[index], self.test_target[index], self.ctrl_expr[index]
        else:
            all_genes = self.basal_ctrl_adata.var.index.to_list()
            de_idx = [all_genes.index(gene) for gene in self.deg_dict[self.perturbation] if gene in all_genes]
            return self.X_test[index], self.test_target[index], {"de_idx": de_idx}, self.ctrl_expr[index]

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        elif self.stage == "test":
            return len(self.X_test)
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train', 'val' or 'test'")
