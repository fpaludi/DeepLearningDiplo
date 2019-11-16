"""Exercise 2

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_2_train_petfinder.py --dataset_dir ./TP01 --epochs 5 --dropout 0.1 0.1 --hidden_layer_sizes 100 60 --experiment_name 'TP01_Ejercicio1_t0'

To know which GPU to use, you can check it with the command

    $ nvidia-smi
"""

import argparse
import os
import mlflow
import pandas
import nltk
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

from itertools import zip_longest
from gensim import corpora
from pprint import pprint
#from nltk import word_tokenize
#from nltk.corpus import stopwords

nltk.download(["punkt", "stopwords"])

TARGET_COL         = 'AdoptionSpeed'
EMBEDDING_SIZE_DIV = 5
MAX_SEQUENCE_LEN   = 55 # Determine in exploratory notebook
EMBEDDINGS_DIM     = 100  # Given by the model (in this case glove.6B.100d)
SW                 = set(nltk.corpus.stopwords.words("english"))

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
    parser.add_argument('--filter_widths', nargs='+', default=[3], type=int,
                        help='CNN filter witdhs.')
    parser.add_argument('--filter_count',  default=16, type=int,
                        help='CNN filter count.')
    parser.add_argument('--batch_size', type=int, default=128,
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


def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    # dataset, dev_dataset = train_test_split(
    #     pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    dataset = pandas.read_csv(os.path.join(dataset_dir, 'train.csv'))
    
    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    #return dataset, dev_dataset, test_dataset
    return dataset, test_dataset


def dataset_generator(ds, one_hot_columns, numeric_columns, embedded_columns, vocabulary, test_data=False):
    if test_data:
        nlabels = None
    else:
        nlabels = ds[TARGET_COL].unique().shape[0]
    
    for n_col in numeric_columns:
        ds["norm_" + n_col] = tf.keras.utils.normalize(ds[n_col].values.reshape(-1,1))
    
    for _, row in ds.iterrows():
        instance = {}
        
        # One hot encoded features
        # Numeric features (should be normalized beforehand)
        # TODO: Add numeric features for row
        instance["direct_features"] = np.hstack([
            tf.keras.utils.to_categorical(row[one_hot_col] - 1, max_value)
            for one_hot_col, max_value in one_hot_columns.items()
        ] + [row["norm_" + n_col] for n_col in numeric_columns])

        # Embedded features
        for embedded_col in embedded_columns:
            instance[embedded_col] = [row[embedded_col]]
        
        # Document to indices for text data, truncated at MAX_SEQUENCE_LEN words
        instance["description"] = vocabulary.doc2idx(
            row["TokenizedDescription"],
            unknown_word_index=len(vocabulary)
        )[:MAX_SEQUENCE_LEN]
        
        # One hot encoded target for categorical crossentropy
        if test_data:
            yield instance
        else:
            target = tf.keras.utils.to_categorical(row[TARGET_COL], nlabels)
            yield instance, target

            
def tokenize_description(description):
    return [w.lower() for w in nltk.word_tokenize(description, language="english") if w.lower() not in SW]


def process_description(dataset):
    # Tokenization
    dataset["TokenizedDescription"] = dataset["Description"].fillna(value="").apply(tokenize_description)
    
    # Import Vocabulary
    vocabulary = corpora.Dictionary(dataset["TokenizedDescription"])
    vocabulary.filter_extremes(no_below=1, no_above=1.0, keep_n=10000)
    
    embeddings_index = {}

    with open("./dataset/glove.6B.100d.txt", "r") as fh:
        for line in fh:
            values = line.split()
            word = values[0]
            if word in vocabulary.token2id:  # Only use the embeddings of words in our vocabulary
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    return vocabulary, embeddings_index
    
    
def create_keras_model(embedded_input, direct_input, hidden_layers, dropouts, embedding_matrix, nlabels, 
                       filter_widths, filter_count, seed=1919):
    tf.keras.backend.clear_session()
    initializer = tf.keras.initializers.glorot_uniform(seed=seed)

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
    direct_features_input_shape = direct_input.shape[0]
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Word embedding layer
    description_input = layers.Input(shape=(MAX_SEQUENCE_LEN,), name="description")
    inputs.append(description_input)

    word_embeddings_layer = layers.Embedding(
        embedding_matrix.shape[0],
        EMBEDDINGS_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LEN,
        trainable=False,
        name="word_embedding"
    )(description_input)
    
    ## TODO: Create a NN (CNN or RNN) for the description input (replace the next)
    DESCRIPTION_FEATURES_LAYER_SIZE = 512
    FILTER_WIDTHS = filter_widths #[3, 5, 7]  # Take 2, 3, and 5 words
    FILTER_COUNT  = filter_count  #16
    
    #description_features = layers.Flatten()(word_embeddings_layer)  # This is a simple concatenation
    #description_features = layers.Dense(
    #units=DESCRIPTION_FEATURES_LAYER_SIZE, 
    #activation="relu", 
    #name="description_features")(description_features)
    
    conv_layers = []
    for filter_width in FILTER_WIDTHS:
        layer = tf.keras.layers.Conv1D(
            FILTER_COUNT,
            filter_width,
            activation="relu",
            name="conv_{}_words".format(filter_width)
        )(word_embeddings_layer)
        layer = tf.keras.layers.GlobalMaxPooling1D(name="max_pool_{}_words".format(filter_width))(layer)
        conv_layers.append(layer)
        
        
    convolved_features = tf.keras.layers.Concatenate(name="convolved_features")(conv_layers)
    description_features = layers.Flatten()(convolved_features)
    description_features_do = layers.Dropout(0.3)(description_features)
    
    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [description_features_do, direct_features_input])
    #features = layers.concatenate(embedding_layers + direct_features_input + conv_layers)
    
    
    # Creating Models
    n_layers = len(hidden_layers)
    if len(dropouts) > n_layers:
        dropouts = dropouts[:n_layers]
        
    for n_neurons, drop, layer in zip_longest(hidden_layers, dropouts, range(n_layers)):
        if layer == 0:
            dense      = layers.Dense(n_neurons, activation='relu',kernel_initializer=initializer)(features)
            last_layer = dense
        else:
            dense      = layers.Dense(n_neurons, activation='relu',kernel_initializer=initializer)(last_layer)
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
    print(np.__version__)
    args = read_args()
    dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
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
    
    # Tensor flow types
    instance_types = {
    "direct_features": tf.float32,
    "description": tf.int32
    }

    for embedded_col in args.embedded_cols:
        instance_types[embedded_col] = tf.int32

    vocabulary, embeddings_index = process_description(dataset)

    
    # ---------------------------------------------
    # Train, Validation, Test and Padding
    # ---------------------------------------------
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(dataset, one_hot_columns, numeric_columns, embedded_columns, vocabulary),
        output_types=(instance_types, tf.int32)
    )

    TRAIN_SIZE = int(dataset.shape[0] * 0.8)
    DEV_SIZE   = dataset.shape[0] - TRAIN_SIZE
    BATCH_SIZE = args.batch_size

    #shuffled_dataset = tf_dataset.shuffle(TRAIN_SIZE + DEV_SIZE, seed=241)

    # Pad the datasets to the max value for all the "non sequence" features
    padding_shapes = (
        {k: [-1] for k in ["direct_features"] + list(embedded_columns.keys())},
        [-1]
    )

    # Pad to MAX_SEQUENCE_LEN for sequence features
    padding_shapes[0]["description"] = [MAX_SEQUENCE_LEN]

    # Pad values are irrelevant for non padded data
    padding_values = (
        {k: 0 for k in list(embedded_columns.keys())},
        0
    )

    # Padding value for direct features should be a float
    padding_values[0]["direct_features"] = np.float32(0)

    # Padding value for sequential features is the vocabulary length + 1
    padding_values[0]["description"] = len(vocabulary) + 1

    train_dataset = tf_dataset.skip(DEV_SIZE).shuffle(TRAIN_SIZE, seed=421)\
        .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values).repeat()

    dev_dataset = tf_dataset.take(DEV_SIZE).shuffle(DEV_SIZE, seed=4322)\
        .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values).repeat()

    # TEST --------------------------------------------
    # First tokenize the description
    test_dataset["TokenizedDescription"] = test_dataset["Description"]\
        .fillna(value="").apply(tokenize_description)

    # Generate the basic TF dataset
    tf_test_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(test_dataset, one_hot_columns, numeric_columns, embedded_columns, vocabulary, True),
        output_types=instance_types  # It should have the same instance types
    )
    
    # Padding
    test_data = tf_test_dataset.padded_batch(
                                            BATCH_SIZE, 
                                            padded_shapes=padding_shapes[0], 
                                            padding_values=padding_values[0]
                                            )
    
    # ------------------------------------------------------
    # Embeddings Word Matrix
    # ------------------------------------------------------
    embedding_matrix = np.zeros((len(vocabulary) + 2, 100))

    for widx, word in vocabulary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[widx] = embedding_vector
        else:
            # Random normal initialization for words without embeddings
            embedding_matrix[widx] = np.random.normal(size=(100,))  

    # Random normal initialization for unknown words
    embedding_matrix[len(vocabulary)] = np.random.normal(size=(100,))
    
    # ------------------------------------------------------
    # Build Model
    # ------------------------------------------------------
    # TODO: Build the Keras model (Ready)
    ldata, ltarget = tf_dataset.take(2)
    #pprint(ldata)
    
    np.random.seed(53221)
    for n_model in range(5):
        local_seed = np.random.randint(1, 20000)

        model = create_keras_model(embedded_columns, ldata[0]['direct_features'], args.hidden_layer_sizes,
                                   args.dropout, embedding_matrix, nlabels, args.filter_widths, args.filter_count,
                                   seed=local_seed)

        #opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

        model.compile(loss='categorical_crossentropy', optimizer="adam",
                  metrics=['accuracy'])
        print(model.summary())
        plot_model(model, to_file='model_cnn.png', show_shapes=True)

        # ------------------------------------------------------
        # Train Model
        # ------------------------------------------------------
        # Weight classes
        class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(dataset.AdoptionSpeed.values),
                                                     dataset.AdoptionSpeed.values)

        # TODO: Fit the model (Ready)
        mlflow.set_experiment(args.experiment_name)
        run_name = args.run_name + "_{}".format(n_model)

        with mlflow.start_run(nested=True, run_name=run_name):
            # Log model hiperparameters first
            mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
            mlflow.log_param('dropout', args.dropout)
            mlflow.log_param('cnn_filter_widths', args.filter_widths)
            mlflow.log_param('cnn_filter_count', args.filter_count)
            mlflow.log_param('dropout', args.dropout)
            mlflow.log_param('embedded_columns', embedded_columns)
            mlflow.log_param('one_hot_columns', one_hot_columns)
            mlflow.log_param('numerical_columns', numeric_columns)
            mlflow.log_param('total_columns', numeric_columns + list(one_hot_columns.keys()) + list(embedded_columns.keys()))
            mlflow.log_param('epochs', args.epochs)
            mlflow.log_param('n_params', model.count_params())
            mlflow.log_param('run_name', args.run_name)

            # Early Stopping
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=5,
                                                  verbose=3, 
                                                  mode='auto',
                                                  restore_best_weights=True)
            # Train
            history = model.fit(train_dataset, epochs=args.epochs,
                                steps_per_epoch=TRAIN_SIZE//BATCH_SIZE,
                                validation_data=dev_dataset, 
                                validation_steps=DEV_SIZE//BATCH_SIZE,
                                verbose=1,
                                callbacks=[es]
                               ) #, class_weight=class_weights)


            # TODO: analyze history to see if model converges/overfits

            # TODO: Evaluate the model, calculating the metrics.
            # Option 1: Use the model.evaluate() method. For this, the model must be
            # already compiled with the metrics.
            #performance = model.evaluate(X_test, y_test) #(Ready)

            #loss, accuracy = 0, 0
            loss, accuracy = model.evaluate(dev_dataset, steps=DEV_SIZE//BATCH_SIZE,)
            print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
            mlflow.log_metric('dev_loss', loss)
            mlflow.log_metric('dev_accuracy', accuracy)
            maxepochs = len(history.history['accuracy'])
            for epoch in range(maxepochs):
                mlflow.log_metric('hist_train_accuracy', value=history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric('hist_val_accuracy', value=history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric('hist_train_loss', value=history.history['loss'][epoch], step=epoch)
                mlflow.log_metric('hist_val_loss', value=history.history['val_loss'][epoch], step=epoch)
        

    # TODO: Convert predictions to classes
    # TODO: Save the results for submission
    print("Predictions")
    predictions = model.predict(test_data)
    test_dataset["AdoptionSpeed"] = predictions.argmax(axis=1)
    test_dataset.to_csv("./submission_cnn.csv", index=False, columns=["PID", "AdoptionSpeed"])
        
    print('All operations completed')

if __name__ == '__main__':
    main()
