from src.efficient_kan import KAN

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import h5py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, f1_score

import os


from functions import apply_bandpass_filter, filter_ecg_signal, resample_ecg_data, set_channels_to_zero, STFT_ECG_all_channels, min_max_normalize


parser = argparse.ArgumentParser(
    prog='Model Name',
    description='What do you want to save your Model as',
    epilog='Name of the model'
)

parser.add_argument('--file_name', default="1_layer_64_minmax_1lead", type=str, help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Initial Learning rate')
# parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
# parser.add_argument('--activation', type=str, default='relu', help='Activation function')
parser.add_argument('--num_repeats', type=int, default=4, help='Number of times to repeat the samples')
parser.add_argument('--n_channels', type=int, default=0, help='Number of empty channels')
parser.add_argument('--gpu', type=int, default=0, help='Number of empty channels')

args = parser.parse_args()

file_name = args.file_name
epochs = args.epochs
initial_learning_rate = args.initial_learning_rate
batch_size = args.batch_size
num_repeats = args.num_repeats
n = args.n_channels

if not os.path.exists(f"./ECG/{file_name}"):
    os.makedirs(f"./ECG/{file_name}")
    print(f"Created directory: ./ECG/{file_name}")

print('Reading Data')
path_to_hdf5 = '...'
# path_to_hdf5 = '/mnt/data13_16T/jim/ECG_data/Brazil/filtered_data20000.hdf5'
hdf5_dset = 'tracings'
path_to_csv = '...'
# path_to_csv = '/mnt/data13_16T/jim/ECG_data/Brazil/filtered_annotations20000.csv'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:, :, 1]
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

# Read the CSV file
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','SB','AF','ST']]
# Get the column names
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)

print(x.shape)
print('Resampling X')
x = resample_ecg_data(x, 400, 500, 4096)
print('Band passing X')
x = apply_bandpass_filter(x)
print('Filtering X')
x = filter_ecg_signal(x)
print('Min_Max X')
x = min_max_normalize(x)
print('Emptying X channels')
x = set_channels_to_zero(x, n)
print('Transforming x')
x = STFT_ECG_all_channels(500, x)
print(x.shape)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42, shuffle=True)

# Resampling Data
indices = np.where(y_train.sum(axis=1) > 0)[0]
sampled_y_train = y_train[indices]
sampled_x_train = x_train[indices]

# Repeat the samples
repeated_y_train = np.repeat(sampled_y_train, num_repeats, axis=0)
repeated_x_train = np.repeat(sampled_x_train, num_repeats, axis=0)

# Concatenate the repeated samples with the original training dataset
y_train = np.concatenate((y_train, repeated_y_train), axis=0)
x_train = np.concatenate((x_train, repeated_x_train), axis=0)

print(x_train.shape, y_train.shape, type(x_train), type(y_train))
print(x_val.shape, y_val.shape, type(x_val), type(y_val))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.X))

    def __getitem__(self, index):
        # Get the input feature and target label for the given index
        idx = self.indices[index]
        x = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        # Convert to PyTorch tensor and return
        return torch.tensor(x), torch.tensor(label)


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)

# Create the train, validation, and test datasets
trainset = MyDataset(x_train, y_train)
valset = MyDataset(x_val, y_val)

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False)



# Define model

model = KAN([33 * 129 * 1, 64, 6])

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Define loss
criterion = nn.BCEWithLogitsLoss()
model.to(device)

val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auroc': [], 'auprc': []}

for epoch in range(epochs):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 33 * 129 * 1).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            d_output = (torch.sigmoid(output)>0.5).flatten().cpu()
            accuracy = (d_output == labels.flatten().cpu()).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 33 * 129 * 1).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += ((torch.sigmoid(output) > 0.5).flatten() == labels.flatten().to(device)).float().mean().item()
            val_predictions.extend(torch.sigmoid(output).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_auroc = roc_auc_score(val_targets, val_predictions)
        val_recall = recall_score(val_targets, (val_predictions > 0.5).astype(int), average='macro')
        val_precision = precision_score(val_targets, (val_predictions > 0.5).astype(int), average='macro')
        val_f1 = f1_score(val_targets, (val_predictions > 0.5).astype(int), average='macro')
        val_auprc = average_precision_score(val_targets, val_predictions)

        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_accuracy)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1'].append(val_f1)
        val_metrics['auroc'].append(val_auroc)
        val_metrics['auprc'].append(val_auprc)

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, AUROC: {val_auroc}, Recall: {val_recall}, Precision: {val_precision}, F1: {val_f1}, AUPRC: {val_auprc}")

    torch.save(model.state_dict(), f"./ECG/{file_name}/model_{epoch+1}.pt")

# Save model
torch.save(model.state_dict(), f"./ECG/{file_name}/model.pt")

# Convert metrics to DataFrame for easy saving to CSV
val_metrics_df = pd.DataFrame(val_metrics)

# Save metrics to CSV
val_metrics_df.to_csv(f"./ECG/{file_name}/val_metrics.csv", index=False)