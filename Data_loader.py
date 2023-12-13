from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import torch

def prepare_data(data_final, hparams):
    """
    Prepares the training, evaluation, and test datasets and loaders.
    
    Parameters:
    - data_final: A DataFrame containing 'embedding' and 'label' columns.
    - hparams: An instance of HParams class containing hyperparameters.
    
    Returns:
    - train_loader: DataLoader for the training dataset.
    - eval_loader: DataLoader for the evaluation dataset.
    - test_loader: DataLoader for the test dataset.
    """
    # Convert DataFrame columns to numpy arrays
    embeddings = np.array(data_final['embedding'].tolist())
    labels = data_final['label'].values

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.22, random_state=42)

    # Convert numpy arrays to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).view(-1,1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    batch_size = hparams.batch_size
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weights = np.array([weights[t] for t in y_train])

    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True) # type: ignore

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader
