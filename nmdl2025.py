import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_data():
    """
    Downloads and loads neural data from Steinmetz 2019.

    Returns:
        alldat (np.ndarray): Combined array of all sessions.
        brain_areas (list): Nested list of brain area acronyms grouped by region.
        brain_region_names (list): Names corresponding to brain area groups.
    """
    urls = [
        "https://osf.io/agvxh/download",
        "https://osf.io/uv3mw/download",
        "https://osf.io/ehmw2/download"
    ]

    all_parts = []

    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            npzfile = np.load(io.BytesIO(response.content), allow_pickle=True)
            all_parts.append(npzfile['dat'])
            print(f"Loaded dataset {i+1} successfully.")
        except Exception as e:
            print(f"Failed to load dataset {i+1} from {url}: {e}")

    # Combine all data parts
    if all_parts:
        alldat = np.concatenate(all_parts, axis=0)
    else:
        alldat = np.array([])

    # Brain areas
    brain_areas = [
        ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],  # visual cortex
        ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"],  # thalamus
        ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],  # hippocampus
        ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP", "TT"],  # non-visual cortex
        ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"],  # midbrain
        ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],  # basal ganglia
        ["BLA", "BMA", "EP", "EPd", "MEA"]  # cortical subplate
    ]

    brain_regions = [
        "visual_cortex",
        "thalamus",
        "hippocampal",
        "non-visual_cortex",
        "midbrain",
        "basal_ganglia",
        "cortical_subplate"
    ]

    return alldat, brain_areas, brain_regions

def get_used_areas_and_regions(dat, brain_areas, brain_regions):
    # Area-to-region mapping
    region_map = {}
    for region, areas in zip(brain_regions, brain_areas):
        for area in areas:
            region_map[area] = region

    # Areas used in the dataset
    used_areas = sorted(np.unique(dat['brain_area']))

    # Regions corresponding to those areas
    used_regions = sorted(set(region_map.get(area, "unknown") for area in used_areas))

    return used_areas, used_regions, region_map

def get_region_spikes(dat, ctx, target_area, region_map, exclude_areas=None, exclude_regions=None, test_size=0.3, random_state=42):
    """
    Extracts and returns spike data for neurons in and out of a target brain area,
    with optional exclusion of specified areas or regions. Splits into train/test sets and moves to device.

    Returns:
        x0_train, x0_test: spike data from non-target areas
        x1_train, x1_test: spike data from target area
    Raises:
        ValueError: if target_area or excluded areas/regions are not present in the dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exclude_areas = exclude_areas or []
    exclude_regions = exclude_regions or []

    # Get unique areas present in the dataset
    dataset_areas = set(np.unique(dat['brain_area']))

    # Error handling
    if target_area not in dataset_areas:
        raise ValueError(f"Target area '{target_area}' not found in dataset areas: {sorted(dataset_areas)}")

    missing_excluded_areas = [a for a in exclude_areas if a not in dataset_areas]
    if missing_excluded_areas:
        raise ValueError(f"Excluded area(s) not in dataset: {missing_excluded_areas}")

    valid_region_names = set(region_map.values())
    missing_excluded_regions = [r for r in exclude_regions if r not in valid_region_names]
    if missing_excluded_regions:
        raise ValueError(f"Excluded region(s) not recognized: {missing_excluded_regions}")

    # Determine areas to exclude based on region names
    excluded_by_region = [area for area, region in region_map.items() if region in exclude_regions]
    all_excluded_areas = set(exclude_areas + excluded_by_region + [target_area])  # exclude target from x0

    # Boolean masks
    mask_target = (dat['brain_area'] == target_area)
    mask_other = ~np.isin(dat['brain_area'], list(all_excluded_areas))

    # Extract and transpose spikes: (neurons, trials, time) -> (time, trials, neurons)
    x0 = np.transpose(dat['spks'][mask_other][:, ctx], (2, 1, 0))
    x1 = np.transpose(dat['spks'][mask_target][:, ctx], (2, 1, 0))

    # Train test split on trials
    n_trials = x0.shape[1]
    train_idx, test_idx = train_test_split(np.arange(n_trials), test_size=test_size, random_state=random_state)

    # Apply split
    x0_train, x0_test = x0[:, train_idx], x0[:, test_idx]
    x1_train, x1_test = x1[:, train_idx], x1[:, test_idx]

    # Convert to PyTorch tensors on device
    x0_train = torch.tensor(x0_train, dtype=torch.float32, device=device)
    x0_test = torch.tensor(x0_test, dtype=torch.float32, device=device)
    x1_train = torch.tensor(x1_train, dtype=torch.float32, device=device)
    x1_test = torch.tensor(x1_test, dtype=torch.float32, device=device)

    return x0_train, x0_test, x1_train, x1_test

def train_poisson_rnn(x0_train, x1_train, x0_test, x1_test, ncomp=12, dropout=0.2, lr=0.005, weight_decay=1e-5, niter=1000, print_every=50, patience=20):
    """
    Initializes and trains an RNN using Poisson loss.

    Parameters:
    - NetClass: The RNN model class
    - x0_train, x1_train: Training inputs and targets
    - x0_test, x1_test: Validation inputs and targets 
    - ncomp: Number of latent components 
    - lr: Learning rate
    - niter: Number of training iterations
    - print_every: Print progress every N epochs
    - patience: Early stopping patience

    Returns:
    - train_losses: List of training loss values
    - val_losses: List of validation loss values
    - z_test: Final prediction on validation set
    """
    class Net(nn.Module):
        def __init__(self, ncomp, NN1, NN2, bidi=True):
            super(Net, self).__init__()

            # play with some of the options in the RNN!
            self.rnn = nn.RNN(NN1, ncomp, num_layers=1, dropout=dropout,
                            bidirectional=bidi, nonlinearity='tanh')
            self.fc = nn.Linear(ncomp, NN2)

        def forward(self, x):
            y = self.rnn(x)[0]

            if self.rnn.bidirectional:
                # if the rnn is bidirectional, it concatenates the activations from the forward and backward pass
                # we want to add them instead, so as to enforce the latents to match between the forward and backward pass
                q = (y[:, :, :ncomp] + y[:, :, ncomp:]) / 2
            else:
                q = y

            # the softplus function is just like a relu but it's smoothed out so we can't predict 0
            # if we predict 0 and there was a spike, that's an instant Inf in the Poisson log-likelihood which leads to failure
            z = F.softplus(self.fc(q), 10)

            return z, q

    # Infer input/output dimensions
    NN1 = x0_train.shape[-1]
    NN2 = x1_train.shape[-1]

    # Initialize network
    net = Net(ncomp, NN1, NN2, bidi=True)
    
    # Initialize final bias as mean firing rate
    net.fc.bias.data[:] = x1_train.mean((0, 1))

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # Poisson loss with clamp to avoid log(0)
    def poisson_loss(lam, spk):
        return lam - spk * torch.log(lam.clamp(min=1e-8))

    train_losses = []
    val_losses = []
    val_loss_best = float('inf')
    patience_counter = 0

    for k in range(niter):
        net.train()
        z_train, _ = net(x0_train)
        train_loss = poisson_loss(z_train, x1_train).mean()

        if k == 0:
            LL_0 = -train_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            z_test, _ = net(x0_test)
            val_loss = poisson_loss(z_test, x1_test).mean()

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {k}")
                print(f"Best val loss: {val_loss_best:.4f} | Current val loss: {val_loss:.4f}")
                break

        if k % print_every == 0:
            print(f"Epoch {k} | Train Loss = {train_loss.item():.4f} | Val Loss = {val_loss.item():.4f}")

    # Evaluation metrics
    LL = -poisson_loss(z_test, x1_test).mean()
    pseudo_r2 = 1 - (LL / LL_0)
    mse = torch.mean((z_test - x1_test) ** 2).item()

    return train_losses, val_losses, z_test, pseudo_r2.detach().numpy(), mse
