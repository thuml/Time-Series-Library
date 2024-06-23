import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
# from kanKANLayer import kanKANLayer
from layers.kan_feedforward import FeedForward, KANLinear

# Define target function
def target_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0.5
    y[mask1] = np.sin(20 * np.pi * x[mask1]) + x[mask1] ** 2
    mask2 = (0.5 <= x) & (x < 1.5)
    y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))
    mask3 = x >= 1.5
    y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])

    # add noise
    noise = np.random.normal(0, 0.2, y.shape)
    y += noise
    
    return y

# Define MLP and kanKAN
class SimpleMLP(nn.Module):
    def __init__(self, dim=128):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        return self.layers(x-1) # centralize the input


def argument_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', default=20001, type=int
    )
    parser.add_argument(
        '-a', '--activation', default=None
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Generate sample data
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1).to(device)
    y_train = torch.tensor(target_function(x_train)).to(device)

    args = argument_parses()
    activation = args.activation
    print("Activation: ", activation)
    # Instantiate models
    kan_model_16 =  FeedForward(1, 16, activation=activation).to(device)
    kan_model_8 = FeedForward(1, 8, activation=activation).to(device)
    kan_model_24 = FeedForward(1, 24, activation=activation).to(device)

    mlp_model_128 = SimpleMLP(dim=128).to(device)
    mlp_model_64 = SimpleMLP(dim=64).to(device)
    mlp_model_256 = SimpleMLP(dim=256).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_kan_8 = torch.optim.Adam(kan_model_8.parameters(), lr=0.01)
    optimizer_kan_16 = torch.optim.Adam(kan_model_16.parameters(), lr=0.01)
    optimizer_kan_24 = torch.optim.Adam(kan_model_24.parameters(), lr=0.01)
    
    optimizer_mlp_64 = torch.optim.Adam(mlp_model_64.parameters(), lr=0.03)
    optimizer_mlp_128 = torch.optim.Adam(mlp_model_128.parameters(), lr=0.03)
    optimizer_mlp_256 = torch.optim.Adam(mlp_model_256.parameters(), lr=0.03)

    kan_8_losses = []
    kan_16_losses = []
    kan_24_losses = []


    mlp_64_losses = []
    mlp_128_losses = []
    mlp_256_losses = []


    # Train the models
    epochs = args.epochs
    for epoch in range(epochs):
        optimizer_kan_8.zero_grad()
        outputs_kan_8 = kan_model_8(x_train)
        loss_kan_8 = criterion(outputs_kan_8, y_train)
        loss_kan_8.backward()
        optimizer_kan_8.step()

        optimizer_kan_16.zero_grad()
        outputs_kan_16 = kan_model_16(x_train)
        loss_kan_16 = criterion(outputs_kan_16, y_train)
        loss_kan_16.backward()
        optimizer_kan_16.step()

        optimizer_kan_24.zero_grad()
        outputs_kan_24 = kan_model_24(x_train)
        loss_kan_24 = criterion(outputs_kan_24, y_train)
        loss_kan_24.backward()
        optimizer_kan_24.step()

        optimizer_mlp_64.zero_grad()
        outputs_mlp_64 = mlp_model_64(x_train)
        loss_mlp_64 = criterion(outputs_mlp_64, y_train)
        loss_mlp_64.backward()
        optimizer_mlp_64.step()

        optimizer_mlp_128.zero_grad()
        outputs_mlp_128 = mlp_model_128(x_train)
        loss_mlp_128 = criterion(outputs_mlp_128, y_train)
        loss_mlp_128.backward()
        optimizer_mlp_128.step()
        
        optimizer_mlp_256.zero_grad()
        outputs_mlp_256 = mlp_model_256(x_train)
        loss_mlp_256 = criterion(outputs_mlp_256, y_train)
        loss_mlp_256.backward()
        optimizer_mlp_256.step()

        if epoch % 100 == 0:
            kan_8_losses.append(loss_kan_8.item())
            kan_16_losses.append(loss_kan_16.item())
            kan_24_losses.append(loss_kan_24.item())

            mlp_64_losses.append(loss_mlp_64.item())
            mlp_128_losses.append(loss_mlp_128.item())
            mlp_256_losses.append(loss_mlp_256.item())

            print('Epoch {}/{}:\n\
                    \tKAN(8) Loss: {},\n\
                    \tKAN(16) Loss: {},\n\
                    \tKAN(24) Loss: {},\n\
                    \tMLP(64) Loss: {},\n\
                    \tMLP(128) Loss: {}, \n\
                    \tMLP(256) Loss: {}'.format(
                        epoch+1, epochs,
                        loss_kan_8,
                        loss_kan_16,
                        loss_kan_24,
                        loss_mlp_64,
                        loss_mlp_128,
                        loss_mlp_256
                ))
        

    # Test the models
    x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
    y_pred_kan_8 = kan_model_8(x_test).detach()
    y_pred_kan_16 = kan_model_16(x_test).detach()
    y_pred_kan_24 = kan_model_24(x_test).detach()

    y_pred_mlp_64 = mlp_model_64(x_test).detach()
    y_pred_mlp_128 = mlp_model_128(x_test).detach()
    y_pred_mlp_256 = mlp_model_256(x_test).detach()
    
    outdata_df_train = {
        'x_train': x_train.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_train': y_train.cpu().numpy().reshape(1, -1)[0].tolist(),
    }
    outdata_df_train = pd.DataFrame(outdata_df_train, columns=outdata_df_train.keys())
    outdata_df_train.to_csv('train_data.csv', index=False)

    outdata_df_test = {
        'x_test': x_test.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_kan_8': y_pred_kan_8.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_kan_16': y_pred_kan_16.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_kan_24': y_pred_kan_24.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_mlp_64': y_pred_mlp_64.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_mlp_128': y_pred_mlp_128.cpu().numpy().reshape(1, -1)[0].tolist(),
        'y_pred_mlp_256': y_pred_mlp_256.cpu().numpy().reshape(1, -1)[0].tolist(),
    }
    outdata_df_test = pd.DataFrame(outdata_df_test, columns=outdata_df_test.keys())
    outdata_df_test.to_csv('test_data.csv', index=False)

    outloss_df = {
        'kan_8_loss': kan_8_losses,#.cpu().numpy().tolist(),
        'kan_16_loss': kan_16_losses, #.cpu().numpy().tolist(),
        'kan_24_loss': kan_24_losses, #.cpu().numpy().tolist(),
        'mlp_64_loss': mlp_64_losses, #.cpu().numpy().tolist(),
        'mlp_128_loss': mlp_128_losses, #.cpu().numpy().tolist(),
        'mlp_256_loss': mlp_256_losses, #.cpu().numpy().tolist()
    }

    outloss_df = pd.DataFrame(outloss_df, columns=outloss_df.keys())
    outloss_df.to_csv('loss_data.csv', index=False)
