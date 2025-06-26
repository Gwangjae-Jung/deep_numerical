import  os, yaml
from    typing      import  Optional

import  numpy       as  np
from    tqdm        import  tqdm

from    ...utils_main   import  PathLike, get_time_str
from    ..grf           import  grf




def generate_data__charged_pendulum(
        config_path:    PathLike,
        return_:        bool = False
    ) -> Optional[tuple[np.ndarray, dict]]:
    ##################################################
    #   Basic configurations
    ##################################################
    print("\n< Dataset generator for charged pendulums >\n")
    with open(config_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    _GENERATION         = config['GENERATION']
    _TEMPORAL_DOMAIN    = config['TEMPORAL_DOMAIN']
    _INITIAL_CONDITION  = config['INITIAL_CONDITION']
    _GRF_CONFIG         = config['GRF_CONFIG']
    _PHYSICAL_CONFIG    = config['PHYSICAL_CONFIG']
    
    # Unpacking the variables
    SEED:                   int = _GENERATION['SEED']
    NUM_INSTANCES:          int = _GENERATION['NUM_INSTANCES']
    DATA_PATH:              str = _GENERATION['DATA_PATH']
    DATA_NAME:              str = _GENERATION['DATA_NAME']
    DESCRIPTION:            str = _GENERATION['DESCRIPTION']
    
    DURATION:               float = _TEMPORAL_DOMAIN['DURATION']
    STEP:                   float = _TEMPORAL_DOMAIN['STEP']

    INIT_ANGLE:             float = _INITIAL_CONDITION['ANGLE'] * np.pi
    INIT_ANGULAR_VELOCITY:  float = _INITIAL_CONDITION['ANGULAR_VELOCITY'] * np.pi
    
    GRF_LENGTH_SCALE:       float = _GRF_CONFIG['GRF_LENGTH_SCALE']
    GRF_FACTOR:             float = _GRF_CONFIG['GRF_FACTOR']

    ROD_LENGTH:             float = _PHYSICAL_CONFIG['LENGTH_OF_ROD']
    GRAV_ACC:               float = _PHYSICAL_CONFIG['MEAN_GRAVITATIONAL_ACCELERATION']
    MASS:                   float = _PHYSICAL_CONFIG['MASS_OF_PENDULUM']
    CHARGE:                 float = _PHYSICAL_CONFIG['CHARGE_OF_PENDULUM']
    MEAN_E_FIELD:           float = _PHYSICAL_CONFIG['MEAN_ELECTRIC_FIELD']
        

    assert NUM_INSTANCES > 0
    assert DURATION > 0 and STEP > 0
    assert ROD_LENGTH > 0


    np.random.seed(SEED)
    t_grid       = np.arange(0, DURATION + STEP * 0.5, STEP)
    _full_t_grid = np.arange(0, DURATION + STEP * 0.9, STEP / 2)
        # Used `STEP / 2` for the Runge-Kutta method


    
    
    ##################################################
    # Generate data
    ##################################################
    data = []

    # In what follows, `u` is the angular displacement and `v` is the angular velocity
    # The time derivative of `u`
    def phi1(t: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return v
    # The time derivative of `v`, i.e., the angular acceleration
    def phi2(t: np.ndarray, u: np.ndarray, v: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        coeff_sin = -(f1 / ROD_LENGTH)
        coeff_cos = ((CHARGE * f2) / (MASS * ROD_LENGTH))
        return coeff_sin * np.sin(u) + coeff_cos * np.cos(u)


    for instance_number in tqdm(range(1, NUM_INSTANCES + 1)):
        ##################################################
        #   Generate `f_1` and `f_2`
        #   All tensors should be 1-dimensional
        ##################################################
        _full_f1 = grf(_full_t_grid.size, DURATION, GRF_LENGTH_SCALE)
        _full_f2 = grf(_full_t_grid.size, DURATION, GRF_LENGTH_SCALE)
        _full_f1 = GRF_FACTOR * _full_f1 + GRAV_ACC / ROD_LENGTH
        _full_f2 = GRF_FACTOR * _full_f2 + MEAN_E_FIELD
        f1, f1_half_forward = _full_f1[::2], _full_f1[1::2]
        f2, f2_half_forward = _full_f2[::2], _full_f2[1::2]
        
        
        ##################################################
        #   Runge-Kutta method
        #   Method: Order 4
        ##################################################
        u_curr, v_curr = INIT_ANGLE, INIT_ANGULAR_VELOCITY
        u_arr = [u_curr]
        for idx, t_next in enumerate(t_grid[1:]):
            # Variables
            t_curr = t_grid[idx]
            f1_curr, f1_mid, f1_next = f1[idx], f1_half_forward[idx], f1[idx + 1]
            f2_curr, f2_mid, f2_next = f2[idx], f2_half_forward[idx], f2[idx + 1]
            # k1
            k1u = phi1(t_curr, u_curr, v_curr)
            k1v = phi2(t_curr, u_curr, v_curr, f1_curr, f2_curr)
            # k2
            k2u = phi1(t_curr + STEP / 2, u_curr + STEP * k1u / 2, v_curr + STEP * k1v / 2)
            k2v = phi2(t_curr + STEP / 2, u_curr + STEP * k1u / 2, v_curr + STEP * k1v / 2, f1_mid, f2_mid)
            # k3
            k3u = phi1(t_curr + STEP / 2, u_curr + STEP * k2u / 2, v_curr + STEP * k2v / 2)
            k3v = phi2(t_curr + STEP / 2, u_curr + STEP * k2u / 2, v_curr + STEP * k2v / 2, f1_mid, f2_mid)
            # k4
            k4u = phi1(t_curr + STEP, u_curr + STEP * k3u, v_curr + STEP * k3v)
            k4v = phi2(t_curr + STEP, u_curr + STEP * k3u, v_curr + STEP * k3v, f1_next, f2_next)
            # Update
            u_curr = u_curr + STEP * (k1u + 2 * k2u + 2 * k3u + k4u) / 6
            v_curr = v_curr + STEP * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            u_arr.append(u_curr)
            
        
        ##################################################
        #   Save the data
        ##################################################
        f1      = f1.reshape(-1, 1)
        f2      = f2.reshape(-1, 1)
        u_arr   = np.array(u_arr).reshape(-1, 1)
        
        curr_data = np.hstack([f1, f2, u_arr])
        data.append(curr_data)
        



    ##################################################
    #   Merge and save the data
    ##################################################
    print("Stacking the data...")
    data = np.array(data)
    print("\t* Done")


    print("Saving the data...")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    np.savez(
                DATA_PATH + f"/{DATA_NAME}_{get_time_str()}.npz",
                data            = data,
                config          = np.array(config, dtype = np.object_),
                description     = np.array(DESCRIPTION, dtype = np.object_),
            )
    print("\t* Done")
    
    if return_:
        return (data, config)
    else:
        return None


##################################################
##################################################
