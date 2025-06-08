import os
import numpy as np
import pandas as pd
from collections import Counter
import argparse
from tqdm import tqdm
import wandb
import inspect

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from dataclasses import asdict
@dataclass
class TrainConfig:
    wandb_project = 'chemberta'
    wandb_run_name = 'run-pl' # 'run' + str(time.time())
    wandb_entity: str = "ailab" # Indicate the entity to log to in wandb
    wandb_host = 'http://wandb.artfacestudio.com'
    # wandb_key = 
    # "ddp", "ddp_find_unused_parameters_true", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload",
    pl_strategy = 'ddp'
    pl_precision = 'bf16-mixed'
    # train
    num_epochs = 10
    batch_size = 16 # 32는 OOM
    learning_rate = 4e-5
    beta1 = 0.9
    beta2 = 0.95
    # model
    model_id = 'seyonec/PubChem10M_SMILES_BPE_450k'
    
class ClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_sequence_length=128):
        self.texts = dataframe['text'].astype(str).tolist()
        self.labels = dataframe['labels'].tolist()
        self.encoding = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encoding['input_ids'][idx],
            'attention_mask': self.encoding['attention_mask'][idx],
            'labels': self.labels[idx]
        }


class Trainer():
    def __init__(self, fabric, cfg, train_df, valid_df=None, test_df=None):
        self.fabric = fabric
        self.cfg = cfg
        self.train_df = train_df
        self.val_df = valid_df
        self.test_df = test_df

        try:
            self.logger = self.fabric.logger
        except Exception as e:
            print(e)

        self.init_weight_dtype()
        self._build_model()
        self._build_dataloader()
        self.setup_fabric()

    def init_weight_dtype(self):
        precision_str = self.cfg.pl_precision
        if '16' in precision_str or 'transformer-engine' in precision_str:
            if 'bf' in precision_str:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32
    def _build_model(self):
        self.model = RobertaForSequenceClassification.from_pretrained(self.cfg.model_id, num_labels=2)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.cfg.model_id)
        self.fill_mask_pipeline = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer)

        parms = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [
            {'params': parms, 'lr': self.cfg.learning_rate},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.fabric.device == 'cuda'
        extra_args = {'fused': True} if use_fused else dict()
        self.optimizer = torch.optim.AdamW(param_groups, betas=(self.cfg.beta1, self.cfg.beta2), **extra_args)

    def _build_dataloader(self):
        self.trainset = ClassificationDataset(self.train_df, self.tokenizer)
        self.valset = ClassificationDataset(self.val_df, self.tokenizer) if self.val_df is not None else None
        self.testset = ClassificationDataset(self.test_df, self.tokenizer) if self.test_df is not None else None

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4
        )

        if valid_df is not None:
            self.val_loader = DataLoader(
                self.valset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=4
            )

        if test_df is not None:
            self.test_loader = DataLoader(
                self.testset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=4
            )
    def setup_fabric(self):
        if self.train_loader is not None:
            self.train_loader = self.fabric.setup_dataloaders(self.train_loader)

        if self.val_loader is not None:
            self.val_loader = self.fabric.setup_dataloaders(self.val_loader)

        if self.test_loader is not None:
            self.test_loader = self.fabric.setup_dataloaders(self.test_loader)

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def train(self):
        global_step = 0
        self.model.train()
        for epoch in range(self.cfg.num_epochs):
            total_loss = 0
            for batch in self.train_loader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()
                total_loss += loss.item()

                self.fabric.log("train_loss", loss.item(), step=global_step)
                global_step += 1

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.train_loader)}")

            self.model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for batch in self.val_loader:
                    outputs = self.model(**batch)
                    val_loss = outputs.loss
                    print(f"Validation Loss: {val_loss.item()}, epoch: {epoch + 1}")
                    self.fabric.log("val_loss", val_loss.item(), step=global_step)

from rdkit import Chem
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False
    
def filter_invalid_smiles(df, smiles_column='text'):
    return df[df[smiles_column].apply(is_valid_smiles)].reset_index(drop=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    cfg = TrainConfig()

    wandb.login(key=cfg.wandb_key, host=cfg.wandb_host)
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=asdict(cfg)
    )

    fabric = Fabric(
        accelerator='cuda',
        num_nodes=args.nnodes,
        devices=args.ngpus,
        strategy=cfg.pl_strategy,
        precision=cfg.pl_precision,
        loggers=wandb_logger
    )
    fabric.launch()

    tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset("clintox", tasks_wanted=None)

    train_df = filter_invalid_smiles(train_df)
    valid_df = filter_invalid_smiles(valid_df)
    test_df = filter_invalid_smiles(test_df)

    trainer = Trainer(fabric, cfg, train_df, valid_df, test_df)
    trainer.train()

    # for i, batch in enumerate(trainer.train_loader):
    #     input_ids = batch['input_ids'].to(trainer.fabric.device)
    #     attention_mask = batch['attention_mask'].to(trainer.fabric.device)
    #     labels = batch['labels'].to(trainer.fabric.device)

    #     print(input_ids.shape) # (16, 128)
    #     print(attention_mask.shape) # (16, 128)
    #     print(labels.shape) # (16)

    #     break