import tensorflow as tf
import numpy
import os
import argparse
import json

VH_OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', '.outputs/')
VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs/')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    return parser.parse_args()

def logMetadata(epoch, logs):
    print()
    print(json.dumps({
        'epoch': epoch,
        'loss': str(logs['loss']),
        'acc': str(logs['accuracy'])
    }))

args = parse_args()

mnist = tf.keras.datasets.mnist

mnist_file_path = os.path.join(VH_INPUTS_DIR, 'mnist/mnist.npz')

with numpy.load(mnist_file_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

metadataCallback = tf.keras.callbacks.LambdaCallback(on_epoch_end=logMetadata)
model.fit(x_train, y_train, epochs=args.epoch, callbacks=[metadataCallback])

save_path = os.path.join(VH_OUTPUTS_DIR, 'model.h5')
model.save(save_path)
