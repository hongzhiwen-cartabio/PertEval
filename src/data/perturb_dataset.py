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
        embs = anndata.read_h5ad(f'./data/{self.data_name}_{self.fm}.h5ad')
        ctrl_emb = embs.obsm['ctrl_emb']
        pert_emb = embs.obsm['pert_emb']

    # 2. 获取扰动列表并划分训练测试集
        pert_list = embs.obs['guide_ids'].unique()
        with open(f'./data/{self.data_name}_fold_indices.json') as f:
            splits = json.load(f)
        pert_names = pd.read_csv(f'./data/{self.data_name}_fold_names.csv').values[:, 0]
        train_perts = pert_names[splits[self.spectral_parameter]['train_indices']]
        train_perts, valid_perts = train_test_split(train_perts, test_size=0.2, random_state=42)
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
        train_target = torch.FloatTensor(adata[embs.obs['guide_ids'].isin(train_perts)].X.toarray())
        val_target = torch.FloatTensor(adata[embs.obs['guide_ids'].isin(valid_perts)].X.toarray())
        test_target = torch.FloatTensor(adata[embs.obs['guide_ids'].isin(test_perts)].X.toarray())
        assert train_target.shape[0] == X_train.shape[0] and X_test.shape[0] == test_target.shape[0], (str(train_target.shape[0])+','+str(X_train.shape[0]))

    # 5. 保存预处理数据
    #     with gzip.open(f"{feature_path}/train_data.pkl.gz", "wb") as f:
    #         pkl.dump((X_train, train_target), f)
    #     with gzip.open(f"{feature_path}/test_data.pkl.gz", "wb") as f:
    #         pkl.dump((X_test, test_target), f)
    
        return X_train, train_target, X_val, val_target, X_test, test_target, None, None


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
            return self.X_train[index], self.train_target[index]#, self.ctrl_expr[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index]#, self.ctrl_expr[index]
        elif self.stage == "test" and self.deg_dict is None:
            if self.all_perts_test is not None:
                return self.X_test[index], self.test_target[index]#, self.all_perts_test[index]#, self.ctrl_expr[index]
            else:
                return self.X_test[index], self.test_target[index]#, self.ctrl_expr[index]
        else:
            all_genes = self.basal_ctrl_adata.var.index.to_list()
            # de_idx = [all_genes.index(gene) for gene in self.deg_dict[self.perturbation] if gene in all_genes]
            return self.X_test[index], self.test_target[index]#, {"de_idx": de_idx}#, self.ctrl_expr[index]

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        elif self.stage == "test":
            return len(self.X_test)
        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train', 'val' or 'test'")
