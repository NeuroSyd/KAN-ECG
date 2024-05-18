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


from functions import apply_bandpass_filter, filter_ecg_signal, resample_ecg_data, set_channels_to_zero, STFT_ECG_all_channels, min_max_normalize


parser = argparse.ArgumentParser(
    prog='Model Name',
    description='What do you want to save your Model as',
    epilog='Name of the model'
)

parser.add_argument('--file_name', default="1_layer_64_minmax", type=str, help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Initial Learning rate')
# parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
# parser.add_argument('--activation', type=str, default='relu', help='Activation function')
parser.add_argument('--num_repeats', type=int, default=4, help='Number of times to repeat the samples')
parser.add_argument('--n_channels', type=int, default=0, help='Number of empty channels')

args = parser.parse_args()

file_name = args.file_name
epochs = args.epochs
initial_learning_rate = args.initial_learning_rate
# dr = args.dropout_rate
batch_size = args.batch_size
# activation = args.activation
num_repeats = args.num_repeats
n = args.n_channels

# Data section
print('Reading Data')
path_to_hdf5 = '...'
hdf5_dset = 'tracings'
path_to_csv = '...'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:]

# Read the CSV file
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','AF']]
# Get the column names
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)

label_model = ['1dAVb','RBBB','LBBB','SB','AF','ST']


# print('Resampling X')
# x = resample_ecg_data(x, 400, 500, 4096)
print('Band passing X')
x = apply_bandpass_filter(x)
print('Filtering X')
x = filter_ecg_signal(x)
# print(x.shape)
print('Min_Max X')
x = min_max_normalize(x)
print('Emptying X channels')
x = set_channels_to_zero(x, n)

print('Transforming x')
x = STFT_ECG_all_channels(500, x)

print(x.shape, y.shape)

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
valset = MyDataset(x, y)

# Dataloaders
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False)

# Define model
# model = KAN([33 * 129 * 12, 128, 6])
model = KAN([33 * 129 * 12, 64, 6])
# model = KAN([33 * 129 * 12, 32, 32, 6])
# model = KAN([33 * 129 * 12, 16, 16, 16, 16, 6])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Define loss
criterion = nn.BCEWithLogitsLoss()

model_path = f"./ECG/{file_name}/model.pt"  # Adjust the path to your saved model file
model.load_state_dict(torch.load(model_path))

model.to(device)



# Assuming 'model' is your trained model instance
model.eval()

# Initialize dictionaries to store metrics
metrics_dict = {'1dAVb': {}, 'RBBB': {}, 'LBBB': {}, 'AF': {}}

with torch.no_grad():
    val_predictions = []
    val_targets = []

    for images, labels in valloader:
        images = images.view(-1, 33 * 129 * 12).to(device)
        output = model(images)

        # Convert output to probabilities
        probabilities = torch.sigmoid(output).cpu().numpy()

        val_predictions.extend(probabilities)
        val_targets.extend(labels.cpu().numpy())

    val_predictions = np.array(val_predictions)[:, [0,1,2,4]]
    val_targets = np.array(val_targets)

    # Calculate metrics for each class
    for i, class_name, a in [(0, '1dAVb', 0), (1, 'RBBB', 1), (2, 'LBBB', 2), (3, 'AF', 3)]:
        class_targets = val_targets[:, a]

        recall = recall_score(class_targets, (val_predictions[:, i] > 0.5).astype(int))
        precision = precision_score(class_targets, (val_predictions[:, i] > 0.5).astype(int))
        f1 = f1_score(class_targets, (val_predictions[:, i] > 0.5).astype(int))
        auroc = roc_auc_score(class_targets, val_predictions[:, i])
        auprc = average_precision_score(class_targets, val_predictions[:, i])

        # Store metrics in dictionary
        metrics_dict[class_name]['recall'] = recall
        metrics_dict[class_name]['precision'] = precision
        metrics_dict[class_name]['f1'] = f1
        metrics_dict[class_name]['auroc'] = auroc
        metrics_dict[class_name]['auprc'] = auprc

    avg_recall = recall_score(val_targets.flatten(), (val_predictions.flatten()> 0.5).astype(int))
    avg_precision = precision_score(val_targets.flatten(), (val_predictions.flatten()> 0.5).astype(int))
    avg_f1 = f1_score(val_targets.flatten(), (val_predictions.flatten()> 0.5).astype(int))
    avg_auroc = roc_auc_score(val_targets.flatten(), val_predictions.flatten())
    avg_auprc = average_precision_score(val_targets.flatten(), val_predictions.flatten())

    # Store average metrics in dictionary
    metrics_dict['average'] = {'recall': avg_recall, 'precision': avg_precision, 'f1': avg_f1,
                               'auroc': avg_auroc, 'auprc': avg_auprc}

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')

# Save metrics to CSV
metrics_df.to_csv(f"./ECG/{file_name}/CPSC_summary_{n}.csv")
