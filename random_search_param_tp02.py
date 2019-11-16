import numpy as np

# Random search
num_col = ["Age", "Fee"]
oh_l    = ['Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
emb_l   = ["Breed1"]
run_n   = 0

root     = "CUDA_VISIBLE_DEVICES=2 python practico_2_train_petfinder.py --dataset_dir ./TP01 --epochs 50 "
exp_name = " --experiment_name 'random_search_tp02' "
np.random.seed(23211)

min_neurons = 16
for n_models in range(16):
    n_layers  = np.random.randint(3, 6)
    h_layers = []
    for idx in range(n_layers):
        if idx == 0:
            layer = np.random.randint(128, 256 + 128)
        else:
            layer = np.random.randint(min_neurons, h_layers[-1]) if h_layers[-1] > min_neurons else min_neurons
        h_layers.append(layer)
        if layer == min_neurons:
            n_layers = len(h_layers)
            break

    n_cnn_layers  = np.random.randint(1, 5)
    cnn_layers = []
    for idx in range(n_cnn_layers):
        layer = np.random.randint(2, 9)
        while np.isin(layer, np.array(cnn_layers)):
            layer = np.random.randint(2, 9)
        cnn_layers.append(layer)
    cnn_count = np.random.choice([16, 24, 32, 40, 48, 56, 64])

    dropout   = np.random.randint(35, 50, size=n_layers) / 100.

    drop_str  = " --dropout " + " ".join([str(x) for x in dropout])
    layer_str = " --hidden_layer_sizes " + " ".join([str(x) for x in h_layers])
    num_str   = " --numeric_cols " + " ".join(num_col)
    oh_str    = " --one_hot_cols " + " ".join(oh_l)
    emb_str   = " --embedded_cols " + " ".join(emb_l)
    fw        = " --filter_widths " + " ".join([str(x) for x in cnn_layers])
    fc        = " --filter_count {}".format(cnn_count)
    run       = " --run_name 'run{}'".format(run_n)
    run_n    += 1
    instruccion = root + drop_str + layer_str + num_str + oh_str + emb_str + fc + fw + exp_name + run
    print(instruccion)

exit()
