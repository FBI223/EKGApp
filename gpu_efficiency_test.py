import tensorflow as tf
import time

print("✅ TF Version:", tf.__version__)
print("✅ GPU Devices:", tf.config.list_physical_devices('GPU'))



x = tf.random.normal([1000, 1000])
start = time.time()
for _ in range(1000):
    _ = tf.matmul(x, x)
end = time.time()
print("Time:", end - start)
