# %% [markdown]
# ## 1. Set up the experiment

# %% [markdown]
# ### 1-1. Import modules

# %%
import  os, time
from    pathlib             import  Path
from    tqdm.notebook       import  tqdm
import  pickle

import  numpy                       as  np
import  torch
from    torch                       import  nn, optim
from    torch_geometric.data        import  Data
from    torch_geometric.loader      import  DataLoader

import  yaml

from    custom_modules.utils                import  get_time_str
from    custom_modules.utils                import  GaussianNormalizer, npzReader
from    custom_modules.utils                import  GridGenerator
from    custom_modules.pytorch.neuralop     import  GNOLite
from    custom_modules.pytorch.torch_utils  import  count_parameters


time_str = get_time_str()

# %% [markdown]
# ### 1-2. Load the configurations

# %%
with open("config_train.yaml") as f:
    config = yaml.load(f, Loader = yaml.FullLoader)
    _exp   = config['experiment']
    _data  = config['pde_dataset']
    _graph = config['graph']
    _gno   = config['gno']

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
NUM_NODES       = GRID ** 2


RADIUS_TRAIN    = _graph['radius']
SAMPLE_SIZE     = _graph['sample_size']
NUM_SAMPLING    = _graph['num_sampling']

# %% [markdown]
# ## 2. Preprocess data

# %% [markdown]
# ### 2-1. Instantiate the storages

# %%
train_data: dict[str, torch.Tensor]= {
    'coeff':    None,
    'Kcoeff':   None,
    'Kcoeff_x': None,
    'Kcoeff_y': None,
    'sol':      None,
}
val_data: dict[str, torch.Tensor]= {
    'coeff':    None,
    'Kcoeff':   None,
    'Kcoeff_x': None,
    'Kcoeff_y': None,
    'sol':      None,
}


normalizer: dict[str, GaussianNormalizer] = {
    'coeff':    None,
    'Kcoeff':   None,
    'Kcoeff_x': None,
    'Kcoeff_y': None,
    'sol':      None,
}

# %% [markdown]
# ### 2-2. Load the train data

# %%
# Train data
reader = npzReader(TRAIN_PATH)
for cnt, k in tqdm(enumerate(train_data.keys()), desc = "Preprocessing the train data"):
    # Step 1. Load data
    train_data[k] = torch.from_numpy(reader.get_field(k)[TRAIN_MASK, ::DOWNSAMPLE, ::DOWNSAMPLE])
    train_data[k] = train_data[k].flatten(-1)
    train_data[k] = train_data[k].type(torch.float)
    
    # Step 2. Normalize data
    normalizer[k] = GaussianNormalizer(train_data[k])
    train_data[k] = normalizer[k].encode(train_data[k])


# Validation data
for cnt, k in tqdm(enumerate(val_data.keys()), desc = "Preprocessing the validation data"):
    # Step 1. Load data
    val_data[k] = torch.from_numpy(reader.get_field(k)[VAL_MASK, ::DOWNSAMPLE, ::DOWNSAMPLE])
    val_data[k] = val_data[k].flatten(-1)
    val_data[k] = val_data[k].type(torch.float)
    
    # Step 2. Normalize data (NOTE: Uses the normalizers for the train dataset)
    val_data[k] = normalizer[k].encode(val_data[k])

# %% [markdown]
# ### 2-3. Construct graphs

# %%
# NOTE Generate a grid to set the node and edge attributes


grid_generator  = GridGenerator([[0., 1.], [0., 1.]], [GRID, GRID], radius = RADIUS_TRAIN)
grid_full_info  = grid_generator.full_information()
node_index  = grid_full_info['node_index']
edge_index  = grid_full_info['edge_index']
grid        = grid_full_info['grid']

# %%
# NOTE Construct graphs


list_train_data, list_test_data = [], []


for idx in tqdm(range(TRAIN_SIZE)):
    _coeff      = train_data[ 'coeff'  ][idx].reshape(NUM_NODES, -1)
    _Kcoeff     = train_data[ 'Kcoeff' ][idx].reshape(NUM_NODES, -1)
    _Kcoeff_x   = train_data['Kcoeff_x'][idx].reshape(NUM_NODES, -1)
    _Kcoeff_y   = train_data['Kcoeff_y'][idx].reshape(NUM_NODES, -1)
    # Define the node feature
    _x = torch.hstack([grid, _coeff, _Kcoeff, _Kcoeff_x, _Kcoeff_y])
    # Define the node target
    _y = train_data['sol'][idx].reshape(NUM_NODES, -1)
    # Define the edge feature
    _edge_attr = torch.hstack(
        [
            grid[edge_index[0]],
            grid[edge_index[1]],
            _coeff[edge_index[0]],
            _coeff[edge_index[1]]
        ]
    )
    
    # Append the new graph
    list_train_data.append(
        Data(
            x = _x,
            y = _y,
            edge_index  = edge_index,
            edge_attr   = _edge_attr,
        )
    )


for idx in tqdm(range(VAL_SIZE)):
    _coeff      = val_data[ 'coeff'  ][idx].reshape(NUM_NODES, -1)
    _Kcoeff     = val_data[ 'Kcoeff' ][idx].reshape(NUM_NODES, -1)
    _Kcoeff_x   = val_data['Kcoeff_x'][idx].reshape(NUM_NODES, -1)
    _Kcoeff_y   = val_data['Kcoeff_y'][idx].reshape(NUM_NODES, -1)
    # Define the node feature
    _x = torch.hstack([grid, _coeff, _Kcoeff, _Kcoeff_x, _Kcoeff_y])
    # Define the node target
    _y = val_data['sol'][idx].reshape(NUM_NODES, -1)
    # Define the edge feature
    _edge_attr = torch.hstack(
        [
            grid[edge_index[0]],
            grid[edge_index[1]],
            _coeff[edge_index[0]],
            _coeff[edge_index[1]]
        ]
    )
    
    # Append the new graph
    list_test_data.append(
        Data(
            x = _x,
            y = _y,
            edge_index  = edge_index,
            edge_attr   = _edge_attr,
        )
    )

# %% [markdown]
# ### 2-4. Instantiate dataloaders

# %%
train_loader = DataLoader(list_train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader  = DataLoader(list_test_data,  batch_size = BATCH_SIZE, shuffle = True) 

# %% [markdown]
# ## 3. Train the model

# %% [markdown]
# ### 3-1. Initialize the model and instantiate the loss function and the optimizer

# %%
gno = GNOLite(**_gno).to(DEVICE)
print(f"The number of the parameters in the custom GNO\n>>> {count_parameters(gno)}")

for p in gno.parameters():
    if p.ndim == 1:
        nn.init.zeros_(p)
    else:
        nn.init.xavier_uniform_(p)

criterion = nn.MSELoss(reduction = 'mean')
optimizer = optim.Adam(params = gno.parameters(), lr = _exp['learning_rate'])

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
    gno.train()
    _train_time = time.time()
    train_epoch_loss:  torch.Tensor = 0
    train_epoch_error: torch.Tensor = 0
    for batch in train_loader:
        batch: Data = batch.to(DEVICE)
        
        train_pred = gno.forward(batch.x, batch.edge_index, batch.edge_attr)
        train_loss = criterion.forward(train_pred, batch.y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss = train_epoch_loss + (
            train_loss
        ) * len(batch)
        train_pred  = normalizer['sol'].decode(train_pred)
        batch.y     = normalizer['sol'].decode(batch.y)
        train_epoch_error = train_epoch_error + (
            torch.linalg.norm(train_pred - batch.y) / (1e-8 + torch.linalg.norm(batch.y))
        ) * len(batch)
    _train_time = time.time() - _train_time
    train_history['train_time'] += _train_time
    train_epoch_loss    = train_epoch_loss / TRAIN_SIZE
    train_epoch_error   = train_epoch_error / TRAIN_SIZE
    train_history['train_loss'].append(train_epoch_loss.item())
    train_history['train_error'].append(train_epoch_error.item())
    
    
    # NOTE: Validation
    gno.eval()
    val_epoch_loss:     torch.Tensor = 0
    val_epoch_error:    torch.Tensor = 0
    with torch.no_grad():
        for batch in test_loader:
            batch: Data = batch.to(DEVICE)
            
            val_pred = gno.forward(batch.x, batch.edge_index, batch.edge_attr)
            val_loss = criterion.forward(val_pred, batch.y)
            
            val_epoch_loss      = val_epoch_loss + val_loss * len(batch)
            val_pred = normalizer['sol'].decode(val_pred)
            batch.y  = normalizer['sol'].decode(batch.y)
            val_epoch_error     = val_epoch_error + (
                                        torch.linalg.norm(val_pred - batch.y) / (1e-8 + torch.linalg.norm(batch.y))
                                    ) * len(batch)
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
gno.cpu()

# Save the model
os.makedirs(time_str, exist_ok = True)
torch.save(gno.state_dict(), f"{time_str}/gnolite_darcy{RESOLUTION}_res{GRID}.pth")

# Save the normalizer, which will also be used in prediction
normalizer['sol'].cpu()
torch.save(normalizer, f"{time_str}/gnolite_darcy{RESOLUTION}_res{GRID}_normalizer.pth")

# Save the history
with open(f"{time_str}/gnolite_darcy{RESOLUTION}_res{GRID}.pickle", "wb") as f:
    pickle.dump(train_history, f)

# %% [markdown]
# ## End of file


