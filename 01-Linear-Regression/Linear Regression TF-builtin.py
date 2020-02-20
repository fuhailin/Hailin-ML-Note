import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import initializers as init
from tensorflow import keras
from tensorflow import losses
from tensorflow.keras import layers
from tensorflow.keras import optimizers

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

num_inputs = 2
num_examples = 1000
TRUE_W = [2, -3.4]
TRUE_b = 4.2
features = tf.random.normal((num_examples, num_inputs), stddev=1)
labels = TRUE_W[0] * features[:, 0] + TRUE_W[1] * features[:, 1] + TRUE_b
labels += tf.random.normal(labels.shape, stddev=0.01)

lr = 0.003
batch_size = 10
num_epochs = 10
# 将训练数据的特征和标签组合 # 随机读取小批量
dataset = tfdata.Dataset.from_tensor_slices((features, labels)).shuffle(buffer_size=num_examples).batch(batch_size)
data_iter = iter(dataset)

model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))

loss = losses.MeanSquaredError()
trainer = optimizers.SGD(learning_rate=lr)

for epoch in range(num_epochs):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training=True), y)

        grads = tape.gradient(l, model.trainable_variables)
        trainer.apply_gradients(zip(grads, model.trainable_variables))

    real_loss = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch + 1, real_loss.numpy().mean()))

print('TRUE_W:{0}, learned_W:{1};\nTRUE_b:{2}, learned_b:{3}'.format(TRUE_W, model.get_weights()[0],
                                                                     TRUE_b, model.get_weights()[1]))
