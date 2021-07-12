import tensorflow as tf
import numpy
import valohai

def logMetadata(epoch, logs):
    with valohai.metadata.logger() as logger:
        logger.log("epoch", epoch)
        logger.log("accuracy", logs['accuracy'])
        logger.log("loss", logs['loss'])

my_parameters = {
    'epoch': 5
}

my_inputs = {
    'mnist': 's3://onboard-sample/tf-sample/mnist.npz'
}

valohai.prepare(
    step="train-model",
    image='tensorflow/tensorflow:2.4.1',
    default_parameters=my_parameters,
    default_inputs=my_inputs
)

mnist_file_path = valohai.inputs('mnist').path()

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
model.fit(x_train, y_train, epochs=valohai.parameters('epoch').value, callbacks=[metadataCallback])

save_path = valohai.outputs().path('model.h5')
model.save(save_path)