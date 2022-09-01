from utils.models import fmnist_model, image_vae, image_vae_mnist, mnist_model, dae_fmnist_model, dae_mnist_model
from keras import activations, callbacks, Sequential
import os
import time
import pickle

def get_model(dataname, expname, x_train):
    if dataname == 'mnist' and expname == 'vae':
        cae_model = image_vae_mnist(x_train)
    elif expname == 'vae':
        cae_model = image_vae(x_train)
    elif expname == 'ae':
        if dataname == 'mnist':
            cae_model = mnist_model(x_train)
        elif dataname == 'fashion_mnist':
            cae_model = fmnist_model(x_train)
    elif expname == 'dae':
        if dataname == 'mnist':
            cae_model = dae_mnist_model(x_train)
        elif dataname == 'fashion_mnist':
            cae_model = dae_fmnist_model(x_train)

    return cae_model


def train_nn(x_train, y_train, nn_model, 
             epochs, batch_size, lr, loss, metrics=None, 
             weights_file=None, history_file=None, reset=True):
    # Initialize model
    # nn_model = Model(inputs=input_layer, outputs=output_layer)
    # nn_model.summary()
    # Load weights
    if not reset and weights_file and os.path.isfile(weights_file):
        try:
            print('Weights loaded')
            nn_model.load_weights(weights_file)
        except:
            pass
    # nn_model.compile(loss=loss, optimizer='rmsprop', metrics=metrics)

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    nn_model.compile(loss=loss, optimizer='rmsprop', metrics=metrics)

    # Train model
    start = time.time()
    history = nn_model.fit(x_train, y_train, epochs=epochs, 
                           batch_size=batch_size, validation_split=0.2, verbose=0,
                           callbacks=[saveBestModel])
    end = time.time()
    print(end-start, 'secs')
    # Add learning curve of current training session to history
    if not reset and os.path.isfile(history_file):
        with open(history_file, 'rb') as rf:
            total_history = pickle.load(rf)
    else:
        total_history = None
    if total_history:
        for key in total_history.keys():
            total_history[key] += history.history[key]
    else:
        total_history = history.history
    # Save model
    if weights_file:
        nn_model.save_weights(weights_file)
    if history_file:
        pickle.dump(total_history, open(history_file, 'wb'))
    return nn_model, total_history
