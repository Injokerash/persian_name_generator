
import sys
import os


sys.path.append(os.getcwd())

from Code.DataProcessing.NameDataSet import NameDataSet
from Code.utils.set_seed import set_seed
from Code.utils.torch_utils import calculate_loss_on_batch, calculate_loss_on_data_loader,generate_new_word_from_torch_model
from Code.utils.mlflow_utils import already_ran

import os
import click
import mlflow

import torch as t
from torch import nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import DataLoader

@click.command()
@click.option("--name", type=str, default='MLP' )
@click.option("--padding", type=int, default=2, help="How many steps to look back" )
@click.option("--embedding-dim", type=int, default=2 )
@click.option("--hidden-size", type=int, default=100 )
@click.option("--epochs", type=int, default=20)
@click.option("--learning-rate", type=float, default=.01)
@click.option("--batch-size", type=int, default=2)
@click.option("--seed", type=int, default=2)
@click.option("--data-url", type=str, default='../../../Data/Raw/english_names.txt')
@click.option("--train-size-ratio", type=float, default=.8)
def train_mlp(name: str, padding : int, embedding_dim : int, hidden_size : int, epochs : int, learning_rate : float, batch_size : int, seed : int, data_url : str, train_size_ratio : float):
    mlflow.autolog(disable=True)

    params = {
        'data_url' : data_url,
        'seed' : seed,
        'padding' : padding,
        'embedding_dim' : embedding_dim,
        'hidden_size' : hidden_size,
        'epochs' : epochs,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'train_size_ratio' : train_size_ratio
    }

    existing_run = already_ran(name, params)

    if existing_run:
        print(f"Found existing run for entrypoint={name} and parameters={params}")
        return existing_run

    with mlflow.start_run() as run:
        mlflow.pytorch.autolog()
        set_seed(params['seed'])

        data = NameDataSet(params['data_url'], add_padding=params['padding'])

        input = data.to_numpy()

        train_size = int(params['train_size_ratio'] * len(input))
        valid_size = len(input) - train_size

        train_dataset, valid_dataset = random_split(input, [train_size, valid_size])
        train_dataset = t.Tensor(train_dataset).type(t.LongTensor)
        valid_dataset = t.Tensor(valid_dataset).type(t.LongTensor)
        
        number_of_characters = len(data.characters)

        model = nn.Sequential(
            nn.Embedding(number_of_characters, params['embedding_dim']),
            nn.Flatten(),
            nn.Linear(params['padding']*params['embedding_dim'], params['hidden_size']),
            nn.BatchNorm1d(params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], number_of_characters)
        )

        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

        optimizer = Adam(model.parameters(), lr=params['learning_rate'])

        train_loss_list = []
        valid_loss_list = []

        for epoch in range(params['epochs']):
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            for batch in train_loader:
                loss = calculate_loss_on_batch(model, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with t.no_grad():
                valid_loss = calculate_loss_on_data_loader(model, valid_loader)
                mlflow.log_metric("valid_loss", valid_loss, epoch)
                valid_loss_list.append(valid_loss)
            
                train_loss = calculate_loss_on_data_loader(model, train_loader)
                mlflow.log_metric("train_loss", train_loss, epoch)
                train_loss_list.append(train_loss)

                print(f'epoch : {epoch}, validation loss : {valid_loss}, train_loss : {train_loss}')

        mlflow.pytorch.log_model(model, "model")

        name_samples = ''
        for _ in range(200):
            name = generate_new_word_from_torch_model(model, data.start_character, data.end_character, data.padding, data.ctoi, data.itoc)
            name_samples += name + '\n'
            
        mlflow.log_text(name_samples, "Generated names.txt")


if __name__ == "__main__":
    train_mlp()
