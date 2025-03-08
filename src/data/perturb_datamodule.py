import os
import pickle as pkl
from typing import Any, Dict, Optional
import anndata as ad
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from pertpy import data as scpert_data

from src.data.perturb_dataset import PerturbData

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

with open(f'{ROOT_DIR}/cache/data_dir_cache.txt', 'r') as f:
    DATA_DIR = f.read().strip()


class PertDataModule(LightningDataModule):
    """`LightningDataModule` for perturbation data."""

    def __init__(
        self,
        data_dir: str = '',
        data_name: str = "norman",
        split: float = 0.00,
        replicate: int = 0,
        batch_size: int = 64,
        deg_eval: Optional[str] = None,
        eval_pert: Optional[str] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.deg_dict = None
        self.num_genes = None
        self.num_perts = None
        self.pert_data = None
        self.pertmodule = None
        self.adata = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_name = data_name
        self.deg_eval = deg_eval
        self.eval_pert = eval_pert
        self.fm = kwargs.get("fm", None)

        if isinstance(split, float):
            self.spectral_parameter = f"{split:.2f}_{str(replicate)}"
        elif isinstance(split, str):
            self.spectral_parameter = f"{split}_{str(replicate)}"
        elif isinstance(split, int):
            self.spectral_parameter = split
        else:
            raise ValueError("Split must be a float, int or a string!")

        self.save_hyperparameters(logger=False)
        self.data_path = os.path.join(data_dir, self.data_name)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.data_train: Optional[DataLoader] = None
        self.data_val: Optional[DataLoader] = None
        self.data_test: Optional[DataLoader] = None

        self.load_scpert_data = {
            "norman": "norman_2019_raw",
            "replogle_k562": "replogle_2022_k562_essential",
            "replogle_rpe1": "replogle_2022_rpe1",
        }

        self.batch_size_per_device = batch_size

        # need to call prepare and setup manually to guarantee proper model setup
        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        """Download and preprocess data."""
        if self.data_name in ["norman_1", "norman_2", "replogle_k562", "replogle_rpe1"]:
            if "norman" in self.data_name:
                data_name = "norman"
            else:
                data_name = self.data_name
            if f"{self.load_scpert_data[data_name]}.h5ad" not in os.listdir("data/"):
                scpert_loader = getattr(scpert_data, self.load_scpert_data[data_name])
                scpert_loader()
        else:
            print('Customized data, skip download.')
            pass
            # raise ValueError(f"Data name {self.data_name} not recognized. Choose from: 'norman_1', 'norman_2', "
            #                  f"'replogle_k562', or replogle_rpe1")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and create datasets."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            if 'norman' in self.data_name:
                data_name = "norman"
            else:
                data_name = self.data_name
            if data_name in self.load_scpert_data:
                scpert_loader = getattr(scpert_data, self.load_scpert_data[data_name])
                adata = scpert_loader()
            else:
                adata = ad.read_h5ad(f"./data/{data_name}.h5ad")

            # Initialize datasets using PerturbData
            self.train_dataset = PerturbData(
                adata, self.data_path, self.spectral_parameter, None, self.fm, stage="train"
            )
            self.val_dataset = PerturbData(
                adata, self.data_path, self.spectral_parameter, None, self.fm, stage="val"
            )

            if not self.deg_eval:
                self.test_dataset = PerturbData(
                    adata, self.data_path, self.spectral_parameter, None, self.fm, stage="test"
                )
            else:
                deg_dict = pkl.load(open(f"{self.data_path}/de_test/deg_pert_dict.pkl", "rb"))
                self.test_dataset = PerturbData(
                    adata, self.data_path, self.spectral_parameter, None, self.fm,
                    perturbation=self.eval_pert, deg_dict=deg_dict, stage="test"
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=0,#self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Return the datamodule state."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the datamodule state."""
        pass


if __name__ == "__main__":
    _ = PertDataModule()