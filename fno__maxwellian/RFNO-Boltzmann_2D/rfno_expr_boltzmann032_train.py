from    typing              import  Union

import  os, sys, time, shutil
from    pathlib             import  Path
from    tqdm                import  tqdm
import  pickle
import  yaml
from    copy                import  deepcopy

import  numpy       as  np
import  torch
from    torch.utils.data            import  TensorDataset, DataLoader

path_script = Path(os.getcwd())
path_work   = path_script.parent.parent
sys.path.append(str(path_work))

from    deep_numerical.utils     import  get_time_str
from    deep_numerical.utils     import  GaussianNormalizer
from    deep_numerical.utils     import  count_parameters, initialize_weights
from    deep_numerical.utils     import  relative_error
from    deep_numerical.neuralop  import  RFNO

from    train_utils.train_utils  import  load_data, augment_data_2D, exponential_cosine
from    train_utils.train_utils  import  LossFunctions




time_str = get_time_str()
os.makedirs(f"./{time_str}", exist_ok=True)
CONFIG_FILE = "config_train.yaml"
with open(str( path_script / CONFIG_FILE )) as f:
    config  = yaml.load(f, Loader = yaml.FullLoader)
    _exp    = config['experiment']
    _data   = config['pde_dataset']
    _model  = config['rfno']
DIMENSION = len(_model['n_modes'])

# Save the config files
shutil.copy(CONFIG_FILE, '/'.join([time_str, CONFIG_FILE]))




# NOTE Training and data preprocess
RANDOM_SEED:    int = _exp['seed']
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}")

BATCH_SIZE:     int             = _exp['batch_size']
NUM_EPOCHS:     int             = _exp['num_epochs']
TRAIN_SIZE:     int             = None
VAL_SIZE:       int             = None
LEARNING_RATE:  float           = _exp['learning_rate']
DEVICE:         torch.device    = torch.device(f"cuda:{_exp['cuda_index']}")
print(f"* The device to be used >>> {DEVICE}")

RESOLUTION:     int     = _data['resolution']
PATH_TRAIN:     Path    = Path(_data['path_train'])
PATH_VAL:       Path    = Path(_data['path_val'])

NUM_TIME_STEPS: int = _data['num_time_steps']
VHS_ALPHA = 0.0




k_in    = 'data'
train_data: dict[str, torch.Tensor] = {
    k_in:       None,
}
val_data: dict[str, torch.Tensor] = {
    k_in:       None,
}
normalizer: dict[str, GaussianNormalizer] = {
    k_in:       None,
}




# Train data
_train_data, _train_info = load_data(PATH_TRAIN, 32, VHS_ALPHA, 1)
V_MAX           = _train_info['v_max']
WHERE_CLOSED    = _train_info['v_where_closed']
_train_data['data'] = _train_data['data'][:, :NUM_TIME_STEPS]
# _train_data = augment_data_2D(_train_data)

train_data[k_in]    = _train_data['data']
TRAIN_SIZE  = int(train_data[k_in].shape[0])

print(f"The size of the training dataset >>>", TRAIN_SIZE, sep=' ')
print(f"The shape of the training dataset >>>", train_data[k_in].shape, sep=' ')

# Validation data
_val_data, _ = load_data(PATH_VAL, 32, VHS_ALPHA, 2)
_val_data['data'] = _val_data['data'][:, :NUM_TIME_STEPS]
val_data[k_in]    = _val_data['data']
VAL_SIZE    = int(val_data[k_in].shape[0])
print(f"The size of the validation dataset >>>", VAL_SIZE, sep=' ')
print(f"The shape of the validation dataset >>>", val_data[k_in].shape, sep=' ')
print('-'*50)
print(f"The number of time steps >>>", NUM_TIME_STEPS, sep=' ')


# Normalize data
normalizer[k_in]    = GaussianNormalizer(train_data[k_in])
train_data[k_in]    = normalizer[k_in].encode(train_data[k_in])
val_data[k_in]      = normalizer[k_in].encode(val_data[k_in])

# Save the normalizer, which will also be used in prediction
for k in normalizer.keys():
    normalizer[k].cpu()
torch.save(normalizer, f"{time_str}/rfno_boltzmann{RESOLUTION}_res{RESOLUTION}_normalizer.pth")
for k in normalizer.keys():
    normalizer[k].to(DEVICE)




# Train data
train_dataset   = TensorDataset(train_data[k_in])
train_loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation data
val_dataset   = TensorDataset(val_data[k_in])
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)




model = RFNO(**_model).to(DEVICE)
initialize_weights(model, "xavier normal")
print(f"The number of the parameters in the models\n>>> {count_parameters(model)}")

lf = LossFunctions(dimension=DIMENSION, resolution=RESOLUTION, v_max=V_MAX, v_where_closed=WHERE_CLOSED, device=DEVICE)

optimizer = torch.optim.Adam(params=model.parameters(), lr=_exp['learning_rate'])
lr_lambda = exponential_cosine(period=20, half_life=100)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)




train_history: dict[str, Union[list, float]] = {
    'train_rel_error':      [],
    'train_rel_error_cons': [],
    'val_rel_error':        [],
    'val_rel_error_cons':   [],
    'train_time':           0.0,
}

best_val_rel_error:         float   = None
best_val_rel_error_cons:    float   = None
best_model:         RFNO    = None
best_model_idx:     int     = None

USED_TIME_STEPS:    int = 2

_denominator_train  = TRAIN_SIZE * USED_TIME_STEPS * (NUM_TIME_STEPS - USED_TIME_STEPS)
_denominator_val    = VAL_SIZE * USED_TIME_STEPS * (NUM_TIME_STEPS - USED_TIME_STEPS)
arr_coeff_cons = 0.5 * (1.0-torch.pow(0.9, torch.arange(NUM_EPOCHS, device=DEVICE)))
for k in train_data.keys():
    normalizer[k].to(DEVICE)

elapsed_time = time.time()
for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    # NOTE: Train
    model.train()
    coeff_cons = arr_coeff_cons[epoch-1]
    _train_time = time.time()
    train_rel_error:        float   = 0
    train_rel_error_cons:   float   = 0
    for traj, in train_loader:
        traj:           torch.Tensor = traj.to(DEVICE)
        num_trajectories = len(traj)
        
        for idx in range(NUM_TIME_STEPS - USED_TIME_STEPS):
            # Time-marching (in the scaled space)
            data_prev = traj[:, idx]
            data_curr = traj[:, idx+1]
            data_next = traj[:, idx+2]
            pred_curr = model.forward(data_prev)
            pred_next = model.forward(pred_curr)
            # Descale data
            pred_curr = normalizer[k_in].decode(pred_curr)
            pred_next = normalizer[k_in].decode(pred_next)
            data_curr = normalizer[k_in].decode(data_curr)
            data_next = normalizer[k_in].decode(data_next)
            # Loss 1 - Data-driven loss
            rel_error__data = \
                relative_error(pred_curr, data_curr, p=2).sum() + \
                relative_error(pred_next, data_next, p=2).sum()
            # Loss 2 - Conservation loss
            rel_error__cons = lf.compute_loss_cons(
                [pred_curr, pred_next],
                [data_curr, data_next],
            )
            # Compute the total loss
            loss: torch.Tensor = \
                rel_error__data + coeff_cons * rel_error__cons
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Backup
            train_rel_error         += rel_error__data.item()
            train_rel_error_cons    += rel_error__cons.item()
        
    scheduler.step()
    _train_time = time.time() - _train_time
    train_rel_error         /= _denominator_train
    train_rel_error_cons    /= _denominator_train
    train_history['train_time'] += _train_time
    train_history['train_rel_error'].append(train_rel_error)
    train_history['train_rel_error_cons'].append(train_rel_error_cons)
    
    # NOTE: Validation
    model.eval()
    val_rel_error:      float   = 0
    val_rel_error_cons: float   = 0
    with torch.no_grad():
        for traj, in val_loader:
            traj:           torch.Tensor = traj.to(DEVICE)
            num_trajectories = len(traj)
            
            for idx in range(NUM_TIME_STEPS - USED_TIME_STEPS):
                # Time-marching (in the scaled space)
                data_prev = traj[:, idx]
                data_curr = traj[:, idx+1]
                data_next = traj[:, idx+2]
                pred_curr = model.forward(data_prev)
                pred_next = model.forward(pred_curr)
                # Descale data
                pred_curr = normalizer[k_in].decode(pred_curr)
                pred_next = normalizer[k_in].decode(pred_next)
                data_curr = normalizer[k_in].decode(data_curr)
                data_next = normalizer[k_in].decode(data_next)
                # Loss 1 - Data-driven loss
                rel_error__data = \
                    relative_error(pred_curr, data_curr, p=2).sum() + \
                    relative_error(pred_next, data_next, p=2).sum()
                # Loss 2 - Conservation loss
                rel_error__cons = lf.compute_loss_cons(
                    [pred_curr, pred_next],
                    [data_curr, data_next],
                )
                # Backup
                val_rel_error       += rel_error__data.item()
                val_rel_error_cons  += rel_error__cons.item()
            
    val_rel_error       /= _denominator_val
    val_rel_error_cons  /= _denominator_val
    train_history['val_rel_error'].append(val_rel_error)
    train_history['val_rel_error_cons'].append(val_rel_error_cons)
    
    # NOTE: Save the best model
    if (
            (best_val_rel_error is None or best_val_rel_error_cons is None) or 
            (best_val_rel_error+best_val_rel_error_cons > val_rel_error+val_rel_error_cons)
        ):
        best_val_rel_error      = val_rel_error
        best_val_rel_error_cons = val_rel_error_cons
        best_model      = deepcopy(model)
        best_model_idx  = epoch-1
        
    # NOTE: Report
    print(f"\n[ Epoch {epoch} / {NUM_EPOCHS} | coeff_cons: {coeff_cons:.4e} , lr: {scheduler.get_last_lr()[0]:.4e}]")
    print(
        "# Best model:",
        f"epoch {best_model_idx+1}",
        f"({train_history['val_rel_error'][best_model_idx]:.4e}",
        f"|",
        f"{train_history['val_rel_error_cons'][best_model_idx]:.4e})",
        sep=' ',
    )
    print(
        "* [train] Relative error:",
        f"{train_history['train_rel_error'][-1]:.4e}",
        f"|",
        f"{train_history['train_rel_error_cons'][-1]:.4e}",
        sep=' ',
    )
    print(
        "* [valid] Relative error:",
        f"{train_history['val_rel_error'][-1]:.4e}",
        f"|",
        f"{train_history['val_rel_error_cons'][-1]:.4e}",
        sep=' ',
    )
    print('\n')
        
    if (
            best_val_rel_error < 1e-5 and
            train_history['val_rel_error_cons'][best_model_idx] < 1e-5
        ):
        print(f"Early stopping at epoch {epoch}.")
        break
    
    if epoch%10==0:
        # Save the model
        __dir_checkpoint = Path(f"./checkpoint__{time_str}")
        __name_checkpoint = f"rfno_boltzmann{RESOLUTION}_res{RESOLUTION}_epoch{str(epoch).zfill(len(str(NUM_EPOCHS)))}.pth"
        os.makedirs(__dir_checkpoint, exist_ok=True)
        model.cpu()
        best_model.cpu()
        torch.save(model.state_dict(), __dir_checkpoint/__name_checkpoint)
        torch.save(model.state_dict(), __dir_checkpoint/("best__" + __name_checkpoint))
        model.to(DEVICE)
        best_model.to(DEVICE)
        
elapsed_time = time.time() - elapsed_time
print(f"Elapsed time: {int(elapsed_time)} seconds")

model = best_model




model.cpu()
for k in train_data.keys():
    normalizer[k].cpu()

# Save the model
torch.save(model.state_dict(), f"{time_str}/rfno_boltzmann{RESOLUTION}_res{RESOLUTION}.pth")

# Save the history
with open(f"{time_str}/rfno_boltzmann{RESOLUTION}_res{RESOLUTION}.pickle", "wb") as f:
    pickle.dump(train_history, f)

# Clear the GPU memory
torch.cuda.empty_cache()

# Done
print("Done.")

