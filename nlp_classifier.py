"""
NLP Assessment 

Objective:
Build a model that distinguish between houses and apartments based on the information contained in a json file
"""

### 1. Import dependencies

from typing import Optional
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
import argparse

import pytorch_lightning as pl
pl.seed_everything(10)

# 2. Set Defaults

MODEL_NAME_OR_PATH = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32
BATCH_SIZE_TESTING = 1
MAX_SEQ_LENGTH = 64
NUM_WORKERS = 4
DIM = 16
OPTIMIZER = "Adam"
LOSS = "cross_entropy"
LR = 1e-4


### 3. Create Dataset and DataModule objects
"""
Due to the purpose of this PoC, the stage of analyzing correlation between data using tools like pandas, seaborn has not been realized. 

After a complete study of the inputs and the classification objectives, the hypothesys is that the following categories are the most relevant for the classification:
- description
- features
- title
- bathrooms
- bedrooms
- living_area
- price
- rent_price
- total_area

Base on this, there are three inputs with relevant information in text format (description, features and title), which need to be converted into tensors that compress the context of them. Apart from this, the rest of the inputs are discrete values that need to be normalized in order to build a more robust learning process
"""

class CasafariDataset(Dataset):
  # Custom Pytorch Dataset for preprocessing examples provided in pandas DataFrame format
  # label_map = {0: apartment, 1: house}

  def __init__(self, data:pd.DataFrame, model_name_or_path: str, max_seq_length: int):
    self.data = data
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    self.max_seq_length = max_seq_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    encoding_description = self.tokenizer(row.description,
                                     max_length=self.max_seq_length, 
                                     padding="max_length", 
                                     truncation=True, 
                                     add_special_tokens=True,
                                     return_tensors='pt')
    encoding_description['input_ids'] = encoding_description['input_ids'].flatten()
    encoding_description['attention_mask'] = encoding_description['attention_mask'].flatten()
    encoding_features = self.tokenizer(row.features,
                                  max_length=self.max_seq_length, 
                                  padding="max_length", 
                                  truncation=True, 
                                  add_special_tokens=True,
                                  return_tensors='pt')
    encoding_features['input_ids'] = encoding_features['input_ids'].flatten()
    encoding_features['attention_mask'] = encoding_features['attention_mask'].flatten()
    encoding_title = self.tokenizer(row.title,
                              max_length=self.max_seq_length, 
                              padding="max_length", 
                              truncation=True, 
                              add_special_tokens=True,
                              return_tensors='pt')
    encoding_title['input_ids'] = encoding_title['input_ids'].flatten()
    encoding_title['attention_mask'] = encoding_title['attention_mask'].flatten()
    other = torch.tensor(pd.to_numeric(row[["bathrooms", 
                                            "bedrooms", "living_area", 
                                            "price", "rent_price", "total_area"]]).values).float()
    label = torch.tensor([1.0, 0.0] if row["is_apartment"] > row["is_house"] else [0.0, 1.0], 
                         dtype=torch.long).float()

    return {"description": encoding_description,
            "features": encoding_features,
            "title": encoding_title,
            "other": other,
            "label": label}


class CasafariDataModule(pl.LightningDataModule):

  # LightningDataModule object that splits data and encapsualte training, 
  # validation and test dataloaders enabling cpu/gpu with multiple cores

  def __init__(self, args: argparse.Namespace = None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.model_name_or_path = self.args.get("model_name_or_path", MODEL_NAME_OR_PATH)
    self.max_seq_length = self.args.get("max_seq_length", MAX_SEQ_LENGTH)
    self.batch_size = self.args.get("batch_size", BATCH_SIZE)
    self.num_workers = self.args.get("num_workers", NUM_WORKERS)

  def prepare_data(self):
    # Download and normalize data
    # Load the dataset
    with open('dataset.json') as json_file:
        dataset_json = json.load(json_file)
    
    # Convert JSON to DataFrame Using read_json()
    self.dataframe = pd.DataFrame.from_records(dataset_json)
    # Normalize columns
    cols_to_norm = ["bathrooms", "bedrooms", "living_area", "price", 
                    "rent_price", "total_area"]
    self.dataframe[cols_to_norm] = self.dataframe[cols_to_norm].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

  def setup(self, stage: Optional[str] = None):
    # Split data
    self.train_df, test_df = train_test_split(self.dataframe, test_size=0.2)
    self.val_df, self.test_df = train_test_split(test_df, test_size=0.5)

  def train_dataloader(self):
    return DataLoader(
        dataset=CasafariDataset(data=self.train_df, 
                                model_name_or_path=self.model_name_or_path,
                                max_seq_length=self.max_seq_length),
        batch_size=self.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=self.num_workers
    )

  def val_dataloader(self):
    return DataLoader(
        dataset=CasafariDataset(data=self.val_df, 
                                model_name_or_path=self.model_name_or_path,
                                max_seq_length=self.max_seq_length),
        batch_size=self.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=self.num_workers
    )

  def test_dataloader(self):
    return DataLoader(
        dataset=CasafariDataset(data=self.test_df, 
                                model_name_or_path=self.model_name_or_path,
                                max_seq_length=self.max_seq_length),
        batch_size=BATCH_SIZE_TESTING,
        shuffle=False,
        pin_memory=True,
        num_workers=self.num_workers,
    )

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--max_seq_length", type=int, default=MAX_SEQ_LENGTH, help="Maximum input length to transformer embeddings"
        )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate per forward pass during training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=NUM_WORKERS, help="Number of CPU cores for loading data"
    )

    return parser

### 4. Architecture
"""
Based on the characteristics of the problem, where inputs present different types (discrete and text), a good initial solution could be build a mixed trainable Neural Network fed with the embeddings of the text inputs and the normalized values of the discrete inputs

In posterior improvements, the embeddings can be also trainable and different classifiers can be tested.

Regarding the embedddings, a decent hypothesis is to use transformer models due to the excellent performance in the NLP SOTA tasks (like GLUE and SuperGLUE benchmarks). 
More specifically, in this particular case we are interested in mapping each text input to a common vector space pre-trained. For that purpose, the sentence-transformer methods (available in HuggingFace provide great pre-trained methods with solid results without training)

https://huggingface.co/sentence-transformers/all-mpnet-base-v2 
"""

class CasafariNN(nn.Module):

  # Architecure of the MLP + transformer embeddings

  def __init__(self, args: argparse.Namespace = None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.model_name_or_path = self.args.get("model_name_or_path", MODEL_NAME_OR_PATH)
    self.dim = self.args.get("dim", DIM)
    self.embedding = AutoModel.from_pretrained(self.model_name_or_path)
    for p in self.embedding.parameters():
        p.requires_grad = False
    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(self.embedding.config.hidden_size, self.dim)
    self.fc2 = nn.Linear(self.dim*3 + 6, self.dim*3 + 6)
    self.fc3 = nn.Linear(self.dim*3 + 6, 2)
    self.relu = nn.ReLU()

  def sentence_embeddings(self, model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def forward(self, encoding):
    with torch.no_grad():
      out_description = self.embedding(**encoding["description"])
      out_features = self.embedding(**encoding["features"])
      out_title = self.embedding(**encoding["title"])

    out_description = self.sentence_embeddings(out_description, encoding["description"]['attention_mask'])
    out_features = self.sentence_embeddings(out_features, encoding["features"]['attention_mask'])
    out_title = self.sentence_embeddings(out_title, encoding["title"]['attention_mask'])
    out_description = self.fc1(out_description)
    out_description = self.relu(out_description)
    out_features = self.fc1(out_features)
    out_features = self.relu(out_features)
    out_title= self.fc1(out_title)
    out_title = self.relu(out_title)
    x = torch.cat((out_description, out_features, out_title, encoding["other"]), dim=-1).float()
    x = self.fc2(x)
    x = self.dropout(x)
    x = self.fc3(x)
    output = nn.functional.softmax(x, dim=1)

    return output

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--dim", type=int, default=DIM, help="Dimension of the internal fully connected layers"
    )
    return parser


class CasafariClassifier(pl.LightningModule):

  # Lightning module  for binary classification of houses and apartments
  
  def __init__(self, model, args: argparse.Namespace = None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.model = model
    self.lr = self.args.get("lr", LR)
    optimizer_type = self.args.get("optimizer", OPTIMIZER)
    self.optimizer_class = getattr(torch.optim, optimizer_type)
    loss = self.args.get("loss", LOSS)
    self.loss_fn = getattr(torch.nn.functional, loss)

  def forward(self, encoding):
    return self.model(encoding)

  def step_loss(self, batch):
    logits = self(batch)
    loss = self.loss_fn(logits, batch["label"])
    return loss, logits

  def make_predictions(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    return labels, predictions

  def training_step(self, batch, batch_idx):
    train_loss, _ = self.step_loss(batch)
    self.log("train_loss", train_loss)
    return train_loss
  
  def validation_step(self, batch, batch_idx):
    val_loss, logits = self.step_loss(batch)
    self.log("val_loss", val_loss, prog_bar=True)
    return val_loss

  def test_step(self, batch, batch_idx):
    test_loss, logits = self.step_loss(batch)
    return {"loss": test_loss,
            "predictions": logits,
            "labels": batch["label"]}

  def test_epoch_end(self, outputs):
    labels, predictions = self.make_predictions(outputs)
    precision = precision_score(np.argmax(labels, axis=1).tolist(), 
                                np.argmax(predictions, axis=1).tolist())
    recall = recall_score(np.argmax(labels, axis=1).tolist(), 
                          np.argmax(predictions, axis=1).tolist())
    f1 = f1_score(np.argmax(labels, axis=1).tolist(), 
                  np.argmax(predictions, axis=1).tolist())
    self.log("F1", f1)
    self.log("Recall", recall)
    self.log("Precision", precision)

  def configure_optimizers(self):
    return self.optimizer_class(self.parameters(), lr=self.lr)

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--lr", type=float, default=LR, help="Learning rate of the training step"
    )
    parser.add_argument(
        "--optimizer", type=str, default=OPTIMIZER, help="Optimizer class from torch.optim"
    )
    parser.add_argument(
        "--loss", type=str, default=LOSS, help="Loss function from torch.functional"
    )
    return parser


# 4. Main
def _setup_parser():
  parser = argparse.ArgumentParser(add_help=False)

  # Program level args (embedding model, data_path ...)
  parser.add_argument(
      "--model_name_or_path", type=str, default=MODEL_NAME_OR_PATH, help="Transformer model that build embeddings"
  )
  
  # Model and Datamodule specific args
  parser = CasafariDataModule.add_to_argparse(parser)
  parser = CasafariClassifier.add_to_argparse(parser)
  # Trainer specific args
  parser = pl.Trainer.add_argparse_args(parser)
  args = parser.parse_args()
  return args


def main():
    # Set args
    args = _setup_parser()
    
    # Add callbacks regarding early stopping and model_checkpoint storage
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
      )
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    
    # Create DataModule object
    datamodule = CasafariDataModule(args=args)
    
    # Create Model
    base_model = CasafariNN(args=args)
    model = CasafariClassifier(model=base_model,args=args)
    
    # Create Trainer (run on GPU or CPU)
    trainer = pl.Trainer(accelerator="auto", 
                         logger=pl.loggers.CSVLogger(save_dir="logs/"),
                         callbacks=callbacks, 
                         max_epochs=args.max_epochs)
                         
    # Training and validation
    trainer.fit(model=model, datamodule=datamodule)
    
    # Testing
    trainer.test(model=model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()
