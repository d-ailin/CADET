import tensorflow as tf

from keras.datasets import mnist, fashion_mnist


from tensorflow.keras import backend as K
from pathlib import Path

from utils.data_util import adjust_proportions
from utils.model_util import get_model, train_nn


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)




dataname = 'fashion_mnist'
# dataname='mnist'

expname = 'ae'
# expname = 'vae'

num_grid = 20
data_prop = 1
anom_train_prop = 0.
# set anom_test_prop to 1 to the test proportions: 10% normal, 90% anom
anom_test_prop = 1.



## data preparation
if dataname == 'mnist':
    (img_train, label_train), (img_test, label_test) = mnist.load_data() 
elif 'fashion_mnist' in dataname:
    (img_train, label_train), (img_test, label_test) = fashion_mnist.load_data()
else:
    print('Invalid Dataset')

# Should Only Run Once
img_train =  img_train.reshape((*img_train.shape, 1)) / 255.0
img_test = img_test.reshape((*img_test.shape, 1)) / 255.0


test_indices = range(10)

for i in test_indices:
    print("Normal class is ", i)
    norm_classes = [i]
    anom_classes = [x for x in range(10) if x not in norm_classes]

    x_train, y_train, y_train_org, _ = adjust_proportions(img_train, label_train, 
                                                anom_train_prop, data_prop, 
                                                norm_classes, anom_classes)
    # print(x_train.shape)

    nn_model = get_model(dataname, expname, x_train)

    cae_model, cae_history = train_nn(x_train=x_train, y_train=x_train, 
                      nn_model=nn_model,
                      epochs=500, batch_size=512, lr=1e-3, 
                      loss='mean_squared_error', 
                      weights_file='./saved_models/{}/{}_weights_{}.h5'.format(dataname, expname, norm_classes), 
                      history_file='./saved_models/{}/{}_history_{}.pkl'.format(dataname, expname, norm_classes), 
                      reset=True)


