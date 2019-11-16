"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ./TP01 --epochs 5 --dropout 0.1 0.1 --hidden_layer_sizes 100 60 --experiment_name 'TP01_Ejercicio1_t0'

To know which GPU to use, you can check it with the command

    $ nvidia-smi
"""

import argparse

import os
import mlflow
import pandas
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

from itertools import zip_longest
import collections

TARGET_COL = 'AdoptionSpeed'
EMBEDDING_SIZE_DIV = 5



def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='../petfinder_dataset', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of instances in each batch.')
    parser.add_argument('--one_hot_cols', type=str, nargs='+', default=['Gender', 'Color1'],
                        help='One hot columns.')
    parser.add_argument('--numeric_cols', type=str, nargs='+', default=['Age', 'Fee'],
                        help='Numeric columns.')
    parser.add_argument('--embedded_cols', type=str, nargs='+', default=['Breed1'],
                        help='Embedded columns.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    parser.add_argument('--run_name', type=str, default='run00',
                        help='Name of the run, used in mlflow.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # TODO Create and append numeric columns (Ready)
    # Don't forget to normalize!
    for n_col in numeric_columns:
        direct_features.append(tf.keras.utils.normalize(df[n_col].values.reshape(-1,1)))
    
    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': np.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None
    
    return features, targets



def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    return dataset, dev_dataset, test_dataset


def create_keras_model(embedded_input, direct_input, hidden_layers, dropouts, nlabels, init_seed):
    tf.keras.backend.clear_session()
    initializer = tf.keras.initializers.glorot_uniform(seed=init_seed)


    # Add one input and one embedding for each embedded column
    embedding_layers = []
    inputs           = []
    for embedded_col, max_value in embedded_input.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / EMBEDDING_SIZE_DIV)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input_shape = direct_input.shape[1]
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])
    
    # Creating Models
    n_layers = len(hidden_layers)
    if len(dropouts) > n_layers:
        dropouts = dropouts[:n_layers]
        
    for n_neurons, drop, layer in zip_longest(hidden_layers, dropouts, range(n_layers)):
        if layer == 0:
            dense      = layers.Dense(n_neurons, activation='relu', kernel_initializer=initializer)(features)
            last_layer = dense
        else:
            dense      = layers.Dense(n_neurons, activation='relu', kernel_initializer=initializer)(last_layer)
            last_layer = dense
        if drop is not None:
            #drop_layer = layers.BatchNormalization()(last_layer)
            #last_layer = drop_layer
            drop_layer = layers.Dropout(drop)(last_layer)
            last_layer = drop_layer
    
    output_layer = layers.Dense(nlabels, activation='softmax')(last_layer)
    model        = models.Model(inputs=inputs, outputs=output_layer)            
    return model


def main():
    get_available_gpus()

    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in args.one_hot_cols
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in args.embedded_cols
    }
    numeric_columns = args.numeric_cols
    
    # TODO shuffle the train dataset! (ready)
    from sklearn.utils import shuffle
    dataset = shuffle(dataset, random_state=22)

    # TODO (optional) put these three types of columns in the same dictionary with "column types" (ready)
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    
    # Create the tensorflow Dataset
    batch_size = 64
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    dev_ds   = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds  = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)[0]).batch(batch_size)

    # TODO: Build the Keras model (Ready)
    
    np.random.seed(53221)
    for n_model in range(1):
        local_seed = np.random.randint(1, 20000)
        model = create_keras_model(embedded_columns, X_train['direct_features'], args.hidden_layer_sizes,
                                   args.dropout, nlabels, init_seed=local_seed)

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)

        # Weight classes
        from sklearn.utils import class_weight
        class_weight = class_weight.compute_class_weight('balanced',
                                                     np.unique(dataset.AdoptionSpeed.values),
                                                     dataset.AdoptionSpeed.values)

        # TODO: Fit the model (Ready)
        mlflow.set_experiment(args.experiment_name)
        run_name = args.run_name + "_{}".format(n_model)
        print("Running: ", run_name, )
        with mlflow.start_run(nested=True, run_name=run_name):
            # Log model hiperparameters first
            mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
            mlflow.log_param('dropout', args.dropout)
            mlflow.log_param('embedded_columns', embedded_columns)
            mlflow.log_param('one_hot_columns', one_hot_columns)
            mlflow.log_param('numerical_columns', numeric_columns)
            mlflow.log_param('total_columns', numeric_columns + list(one_hot_columns.keys()) + list(embedded_columns.keys()))
            mlflow.log_param('epochs', args.epochs)
            mlflow.log_param('n_params', model.count_params())
            mlflow.log_param('run_name', args.run_name)

            # Train
            history = model.fit(train_ds, epochs=args.epochs,
                                validation_data=dev_ds, verbose=1, class_weight=class_weight)


            # TODO: analyze history to see if model converges/overfits

            # TODO: Evaluate the model, calculating the metrics.
            # Option 1: Use the model.evaluate() method. For this, the model must be
            # already compiled with the metrics.
            #performance = model.evaluate(X_test, y_test) #(Ready)

            #loss, accuracy = 0, 0
            loss, accuracy = model.evaluate(dev_ds)
            print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
            mlflow.log_metric('dev_loss', loss)
            mlflow.log_metric('dev_accuracy', accuracy)
            for epoch in range(args.epochs):
                mlflow.log_metric('hist_train_accuracy', value=history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric('hist_val_accuracy', value=history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric('hist_train_loss', value=history.history['loss'][epoch], step=epoch)
                mlflow.log_metric('hist_val_loss', value=history.history['val_loss'][epoch], step=epoch)

            # Option 2: Use the model.predict() method and calculate the metrics using
            # sklearn. We recommend this, because you can store the predictions if
            # you need more analysis later. Also, if you calculate the metrics on a
            # notebook, then you can compare multiple classifiers.

            #predictions = 'No prediction yet'
            print("Predictions")
            predictions = model.predict(test_ds)

            # TODO: Convert predictions to classes
            # TODO: Save the results for submission
            # ...
            #print(collections.Counter(predictions))
            test_dataset["AdoptionSpeed"] = predictions.argmax(axis=1)
            test_dataset.to_csv("./submission.csv", index=False, columns=["PID", "AdoptionSpeed"])
        
    print('All operations completed')

if __name__ == '__main__':
    main()
