# %% [markdown]
# ## 1. Set up the experiment

# %% [markdown]
# ### 1-1. Import modules

# %%
import  os, time
from    pathlib             import  Path
from    tqdm.notebook       import  tqdm
import  pickle
import  yaml

import  numpy       as  np
import  torch
from    torch.utils.data            import  TensorDataset, DataLoader

from    custom_modules.utils                import  get_time_str
from    custom_modules.utils                import  GridGenerator, npzReader, GaussianNormalizer
from    custom_modules.pytorch.neuralop     import  FourierNeuralOperatorLite
from    custom_modules.pytorch.torch_utils  import  count_parameters


time_str = get_time_str()

# %% [markdown]
# ### 1-2. Load the configurations

# %%
with open("config_train.yaml") as f:
    config = yaml.load(f, Loader = yaml.FullLoader)
    _exp   = config['experiment']
    _data  = config['pde_dataset']
    _fno   = config['fno']

# %% [markdown]
# ### 1-3. Set the experiment

# %%
# NOTE Training and data preprocess


BATCH_SIZE      = _exp['batch_size']
NUM_EPOCHS      = _exp['num_epochs']
LEARNING_RATE   = _exp['learning_rate']
TRAIN_SIZE      = _exp['train_size']
VAL_SIZE        = _exp['val_size']
DEVICE          = torch.device(f"cuda:{_exp['cuda_index']}")


RESOLUTION      = _data['resolution']
TRAIN_PATH      = Path(_data['path'])
__RANDOM_CHOICE = np.random.choice(1024, TRAIN_SIZE + VAL_SIZE, replace = False)
TRAIN_MASK      = __RANDOM_CHOICE[:TRAIN_SIZE]
VAL_MASK        = __RANDOM_CHOICE[-VAL_SIZE:]


DOWNSAMPLE      = _data['downsample']
GRID            = (RESOLUTION - 1) // DOWNSAMPLE + 1
grid            = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, 1, GRID),
                            torch.linspace(0, 1, GRID),
                            indexing = 'ij'
                        ),
                        dim = -1
                    )   # Shape: (GRID, GRID, dim_domain = 2)

# %% [markdown]
# ## 2. Preprocess data

# %% [markdown]
# ### 2-1. Instantiate the storages

# %%
train_data: dict[str, torch.Tensor] = {
    'grid':     None,
    'coeff':    None,
    'sol':      None,
}
val_data: dict[str, torch.Tensor] = {
    'grid':     None,
    'coeff':    None,
    'sol':      None,
}


normalizer: dict[str, GaussianNormalizer] = {
    'grid':     None,
    'coeff':    None,
    'sol':      None,
}

# %% [markdown]
# ### 2-2. Load the train data

# %%
# Train data
reader = npzReader(TRAIN_PATH)
for cnt, k in tqdm(enumerate(train_data.keys()), desc = "Preprocessing the train data"):   
    # Step 1. Load data
    if k != 'grid':
        train_data[k] = torch.from_numpy(reader.get_field(k)[TRAIN_MASK, ::DOWNSAMPLE, ::DOWNSAMPLE])
        train_data[k] = train_data[k].type(torch.float).to(DEVICE)
    else:
        train_data[k] = grid.clone().unsqueeze(0).repeat(TRAIN_SIZE, 1, 1, 1).to(DEVICE)
    
    # Step 2. Normalize data
    normalizer[k] = GaussianNormalizer(train_data[k])
    normalizer[k].to(DEVICE)
    train_data[k] = normalizer[k].encode(train_data[k])


# Validation data
for cnt, k in tqdm(enumerate(val_data.keys()), desc = "Preprocessing the validation data"):
    # Step 1. Load data
    if k != 'grid':
        val_data[k] = torch.from_numpy(reader.get_field(k)[VAL_MASK, ::DOWNSAMPLE, ::DOWNSAMPLE])
        val_data[k] = val_data[k].type(torch.float).to(DEVICE)
    else:
        val_data[k] = grid.clone().unsqueeze(0).repeat(VAL_SIZE, 1, 1, 1).to(DEVICE)
    
    # Step 2. Normalize data (NOTE: Uses the normalizers for the train dataset)
    val_data[k] = normalizer[k].encode(val_data[k])

# %% [markdown]
# ### 2-3. Merge the grid and the coefficients

# %%
train_data['data'] = torch.cat([train_data['grid'], train_data['coeff']], dim = -1)
val_data['data']   = torch.cat([  val_data['grid'], val_data['coeff']]  , dim = -1)
del(train_data['grid'], train_data['coeff'])
del(  val_data['grid'], val_data['coeff'])

# %% [markdown]
# ### 2-4. Instantiate dataloaders

# %%
train_dataset = TensorDataset(train_data['data'], train_data['sol'])
val_dataset   = TensorDataset(  val_data['data'], val_data['sol'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader   = torch.utils.data.DataLoader(  val_dataset,  batch_size = BATCH_SIZE, shuffle = True) 

# %% [markdown]
# ## 3. Train the model

# %% [markdown]
# ### 3-1. Initialize the model and instantiate the loss function and the optimizer

# %%
fno = FourierNeuralOperatorLite(**_fno).to(DEVICE)
print(f"The number of the parameters in the custom FNO\n>>> {count_parameters(fno)}")
print(fno)

for p in fno.parameters():
    if p.ndim == 1:
        torch.nn.init.zeros_(p)
    else:
        torch.nn.init.xavier_uniform_(p)

criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(params = fno.parameters(), lr = _exp['learning_rate'])

# %% [markdown]
# ### 3-2. Train the model

# %%
train_history = {
    'train_loss':   [],
    'train_error':  [],
    'val_loss':     [],
    'val_error':    [],
    'train_time':   0.0,
}
normalizer['sol'].to(DEVICE)

elapsed_time = time.time()
for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    # NOTE: Train
    fno.train()
    _train_time = time.time()
    train_epoch_loss:  torch.Tensor = 0
    train_epoch_error: torch.Tensor = 0
    for data, target in train_loader:
        num_data = len(data)
        
        train_pred = fno.forward(data)
        train_loss = criterion.forward(train_pred, target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss = train_epoch_loss + train_loss * num_data
        train_pred  = normalizer['sol'].decode(train_pred)
        target      = normalizer['sol'].decode(target)
        train_epoch_error = train_epoch_error + (
            torch.linalg.norm(train_pred - target) / (1e-8 + torch.linalg.norm(target))
        ) * num_data
    _train_time = time.time() - _train_time
    train_history['train_time'] += _train_time
    train_epoch_loss    = train_epoch_loss / TRAIN_SIZE
    train_epoch_error   = train_epoch_error / TRAIN_SIZE
    train_history['train_loss'].append(train_epoch_loss.item())
    train_history['train_error'].append(train_epoch_error.item())
    
    
    # NOTE: Validation
    fno.eval()
    val_epoch_loss:     torch.Tensor = 0
    val_epoch_error:    torch.Tensor = 0
    with torch.no_grad():
        for data, target in val_loader:
            num_data = len(data)
            
            val_pred = fno.forward(data)
            val_loss = criterion.forward(val_pred, target)
            
            val_epoch_loss      = val_epoch_loss + val_loss * num_data
            val_pred = normalizer['sol'].decode(val_pred)
            target   = normalizer['sol'].decode(target)
            val_epoch_error     = val_epoch_error + (
                                        torch.linalg.norm(val_pred - target) / (1e-8 + torch.linalg.norm(target))
                                    ) * num_data
    val_epoch_loss      = val_epoch_loss / VAL_SIZE
    val_epoch_error     = val_epoch_error / VAL_SIZE
    train_history['val_loss'].append(val_epoch_loss.item())
    train_history['val_error'].append(val_epoch_error.item())
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"[ Epoch {epoch} / {NUM_EPOCHS} ]")
        for k in train_history.keys():
            if k == "train_time":
                continue
            print(f"* {k:15s}: {train_history[k][-1]:.4e}")
    
elapsed_time = time.time() - elapsed_time
print(f"Elapsed time: {int(elapsed_time)} seconds")

# %% [markdown]
# ### 3-3. Save the model and the train history

# %%
# Save the model
os.makedirs(f"./{time_str}", exist_ok = True)
torch.save(fno.state_dict(), f"{time_str}/fnolite_darcy{RESOLUTION}_res{GRID}.pth")

# Save the normalizer, which will also be used in prediction
normalizer['sol'].cpu()
torch.save(normalizer, f"{time_str}/fnolite_darcy{RESOLUTION}_res{GRID}_normalizer.pth")

# Save the history
with open(f"{time_str}/fnolite_darcy{RESOLUTION}_res{GRID}.pickle", "wb") as f:
    pickle.dump(train_history, f)

# %% [markdown]
# ## End of file


