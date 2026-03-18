import torch

X_train = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
])

y_train = torch.tensor([0, 1, 0, 1, 0])

X_test = torch.tensor([
    [11.0, 12.0],
    [13.0, 14.0]
])
y_test = torch.tensor([1, 0])

from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(f"Number of samples in train_ds: {len(train_ds)}")

torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_ds,
    batch_size = 2,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)