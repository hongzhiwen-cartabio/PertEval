import torch
import os
import anndata

import numpy as np
import scanpy as sc
import pickle as pkl

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.spectra import get_splits


class PerturbData(Dataset):
    def __init__(self, adata, data_path, spectral_parameter, spectra_params, stage):
        self.data_name = data_path.split('/')[-1]
        self.data_path = data_path
        self.spectral_parameter = spectral_parameter
        self.spectra_params = spectra_params
        self.stage = stage

        # todo: calculate correlation vector for each perturbation

        if self.data_name == "norman":
            single_gene_mask = [True if "+" not in name else False for name in adata.obs['perturbation_name']]
            sg_pert_adata = adata[single_gene_mask, :]
            sg_pert_adata.obs['condition'] = sg_pert_adata.obs['perturbation_name'].replace('control', 'ctrl')

            genes = sg_pert_adata.var.index.to_list()
            genes_and_ctrl = genes + ['ctrl']

            # we remove the cells with perts that are not in the genes because we need gene expression values
            # to generate an in-silico perturbation embedding
            sg_pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'].isin(genes_and_ctrl), :]

            train, test, pert_list = get_splits.spectra(sg_pert_adata,
                                                        self.data_path,
                                                        self.spectra_params,
                                                        self.spectral_parameter
                                                        )

            print(f"Norman dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

            pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]
            all_perts = list(set(pert_adata.obs['condition'].to_list()))
            unique_perts = list(set(pert_list))

            if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
                ctrl_adata = sg_pert_adata[sg_pert_adata.obs['condition'] == 'ctrl', :]
                ctrl_X = ctrl_adata.X.toarray()
                basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
                subset_size = 500

                for cell in tqdm(range(pert_adata.shape[0])):
                    subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                    basal_ctrl_X[cell, :] = subset.mean(axis=0)

                basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

                # noinspection PyTypeChecker
                basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")
                with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
                    pkl.dump(all_perts, f)
            else:
                basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")

            control_genes = basal_ctrl_adata.var.index.to_list()
            pert_genes = pert_adata.var.index.to_list()
            assert control_genes == pert_genes, ("Watch out! Genes in control and perturbation datasets are not the"
                                                 " same, or are not indexed the same.")

            train_perts = [pert_list[i] for i in train]
            test_perts = [pert_list[i] for i in test]

            highly_variable_genes = pert_adata.var_names[adata.var['highly_variable']]
            hv_pert_adata = pert_adata[:, highly_variable_genes]

            train_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(train_perts), :]
            test_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(test_perts), :]

            all_perts_train = train_target.obs['condition'].values
            all_perts_test = test_target.obs['condition'].values

            # perts_idx = {}
            # for pert in all_perts:
            #     perts_idx[pert] = pert_genes.index(pert)

            if not os.path.exists(f"{self.data_path}/pert_corrs.pkl"):
                correlations = np.zeros(basal_ctrl_adata.shape[1])
                all_gene_expression = basal_ctrl_adata.X

                pert_corrs = {}

                for pert in unique_perts:
                    pert_idx = adata.var_names.get_loc(pert)
                    basal_expr_pert = basal_ctrl_adata.X[:, pert_idx].flatten()
                    for i in range(all_gene_expression.shape[1]):
                        correlations[i] = np.corrcoef(basal_expr_pert, all_gene_expression[:, i])[0, 1]
                    pert_corrs[pert] = correlations
            else:
                with open(f"{self.data_path}/pert_corrs.pkl", "rb") as f:
                    pert_corrs = pkl.load(f)

            num_ctrl_cells = basal_ctrl_adata.shape[0]
            num_train_cells = train_target.shape[0]
            num_test_cells = test_target.shape[0]
            num_genes = basal_ctrl_adata.shape[1]

            pert_corr_train = np.zeros((num_train_cells, num_genes))
            for i, pert in enumerate(all_perts_train):
                pert_corr_train[i, :] = pert_corrs[pert]

            pert_corr_test = np.zeros((num_test_cells, num_genes))
            for i, pert in enumerate(all_perts_test):
                pert_corr_test[i, :] = pert_corrs[pert]

            train_input_expr = basal_ctrl_adata[np.random.randint(0, num_ctrl_cells, num_train_cells), :].X.toarray()
            test_input_expr = basal_ctrl_adata[np.random.randint(0, num_ctrl_cells, num_test_cells), :].X.toarray()

            raw_X_train = np.concatenate((train_input_expr, pert_corr_train), axis=1)
            raw_train_target = train_target.X.toarray()

            X_train, X_val, train_targets, val_targets = train_test_split(raw_X_train,
                                                                          raw_train_target,
                                                                          test_size=0.2)
            self.X_train = torch.from_numpy(X_train)
            self.train_target = torch.from_numpy(train_targets)
            self.X_val = torch.from_numpy(X_val)
            self.val_target = torch.from_numpy(val_targets)
            self.X_test = torch.from_numpy(np.concatenate((test_input_expr, pert_corr_test), axis=1))
            self.test_target = torch.from_numpy(test_target.X.toarray())

            # todo: save all the train val and test data so that we don't have to recompute it every time

            print('joe')

        if self.data_name == "replogle_rpe1":
            ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
            self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)

        if self.data_name == "replogle_k562":
            ctrl_adata, pert_adata, train, test, pert_list = self.preprocess_replogle(adata)
            self.featurise_replogle(pert_adata, pert_list, ctrl_adata, train, test)

    def preprocess_replogle(self, adata):
        if not os.path.exists(f"{self.data_path}/{self.data_name}_filtered.h5ad"):
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            adata.write(f"{self.data_path}/{self.data_name}_filtered.h5ad", compression='gzip')
        else:
            adata = sc.read_h5ad(f"{self.data_path}/{self.data_name}_filtered.h5ad")

        adata.obs['condition'] = adata.obs['perturbation'].replace('control', 'ctrl')

        genes = adata.var.index.to_list()
        genes_and_ctrl = genes + ['ctrl']

        # we remove the cells with perts that are not in the genes because we need gene expression values
        # to generate an in-silico perturbation embedding
        sg_pert_adata = adata[adata.obs['condition'].isin(genes_and_ctrl), :]

        ctrl_adata = sg_pert_adata[sg_pert_adata.obs['condition'] == 'ctrl', :]
        pert_adata = sg_pert_adata[sg_pert_adata.obs['condition'] != 'ctrl', :]
        all_perts = list(set(pert_adata.obs['condition'].to_list()))

        if not os.path.exists(f"{self.data_path}/all_perts.pkl"):
            with open(f"{self.data_path}/all_perts.pkl", "wb") as f:
                pkl.dump(all_perts, f)

        train, test, pert_list = get_splits.spectra(sg_pert_adata,
                                                    self.data_path,
                                                    self.spectra_params,
                                                    self.spectral_parameter
                                                    )
        return ctrl_adata, pert_adata, train, test, pert_list

    def featurise_replogle(self, pert_adata, pert_list, ctrl_adata, train, test):
        print(f"{self.data_name} dataset has {len(pert_list)} perturbations in common with the genes in the dataset.")

        all_perts = pert_adata.obs['condition'].to_list()

        if not os.path.exists(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad"):
            ctrl_X = ctrl_adata.X.toarray()
            basal_ctrl_X = np.zeros((pert_adata.shape[0], ctrl_X.shape[1]))
            subset_size = 500

            for cell in tqdm(range(pert_adata.shape[0])):
                subset = ctrl_X[np.random.choice(ctrl_X.shape[0], subset_size), :]
                basal_ctrl_X[cell, :] = subset.mean(axis=0)

            basal_ctrl_adata = anndata.AnnData(X=basal_ctrl_X, obs=pert_adata.obs, var=ctrl_adata.var)

            # noinspection PyTypeChecker
            basal_ctrl_adata.write(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")
        else:
            basal_ctrl_adata = sc.read_h5ad(f"{self.data_path}/basal_ctrl_{self.data_name}_filtered.h5ad")

        control_genes = basal_ctrl_adata.var.index.to_list()
        pert_genes = pert_adata.var.index.to_list()
        assert control_genes == pert_genes, ("Watch out! Genes in control and perturbation datasets are not the"
                                             " same, or are not indexed the same.")

        train_perts = [pert_list[i] for i in train]
        test_perts = [pert_list[i] for i in test]

        sc.pp.highly_variable_genes(pert_adata, n_top_genes=5000)
        highly_variable_genes = pert_adata.var_names[pert_adata.var['highly_variable']]
        hv_pert_adata = pert_adata[:, highly_variable_genes]

        train_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(train_perts), :]
        test_target = hv_pert_adata[hv_pert_adata.obs['condition'].isin(test_perts), :]

        all_perts_train = train_target.obs['condition'].values
        all_perts_test = test_target.obs['condition'].values

        perts_idx = {}
        for pert in all_perts:
            perts_idx[pert] = pert_genes.index(pert)

        num_ctrl_cells = basal_ctrl_adata.shape[0]
        num_train_cells = train_target.shape[0]
        num_test_cells = test_target.shape[0]
        num_genes = basal_ctrl_adata.shape[1]

        pass # continue here

    def __getitem__(self, index):
        if self.stage == "train":
            return self.X_train[index], self.train_target[index]
        elif self.stage == "val":
            return self.X_val[index], self.val_target[index]
        else:
            return self.X_test[index], self.test_target[index]

    def __len__(self):
        if self.stage == "train":
            return len(self.X_train)
        elif self.stage == "val":
            return len(self.X_val)
        else:
            return len(self.X_test)
