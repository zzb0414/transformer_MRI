from torch.utils.data import Dataset

class ksp_transformer_train_dataset(Dataset):
    def __init__(self, input_array, target_array):
        self.inputs = input_array
        self.targets = target_array

    def __len__(self):
        # Tells the DataLoader how many images are in the set
        return len(self.inputs)

    def __getitem__(self, idx):
        # Grabs one 'tray' of data for the DataLoader to batch
        return self.inputs[idx], self.targets[idx]