import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
from tensorflow.keras.applications import inception_v3
"""
报以下错误：
File "D:\Anaconda3\envs\TF2.1\lib\site-packages\tensorflow_core\python\keras\saving\hdf5_format.py", line 651, in load_weights_from_hdf5_group
    original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'
解决方案：
pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/

"""

base_image_path = keras.utils.get_file("sky.jpg", "https://i.imgur.com/aGBdQyK.jpg")
result_prefix = "sky_dream"
"""
如果训练其他图片，上面两行代码换成下面两行代码,其中picture.jpg换成你图片的名字
base_image_path = "F28w3Ac.jpg"
result_prefix = base_image_path.rstrip('.jpg') + "_dream"
"""

#这些是我们尝试最大化激活的层的名称，以及我们试图最大化激活的最终损失的权重。
# 你可以调整这些设置以获得新的视觉效果。
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 2.0,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# 利用这些超参数也可以获得新的视觉效果
step = 0.01  # 梯度上升步长
num_octave = 3
octave_scale = 1.4
iterations = 20
max_loss = 15.0

def preprocess_image(image_path):
    # 函数功能：调整和格式化图片到正确大小的数组
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 函数功能：把NumPy数组转成一张有效图片
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # 撤消Inception v3预处理
    x /= 2.0
    x += 0.5
    x *= 255.0
    # 转换为uint8并裁剪到有效范围[0，255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

"""
    计算模型损失
"""

# 建立一个预训练好ImageNet的Inception V3模型
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# 获取每个“关键”层的符号输出（我们给它们指定了唯一的名称）。
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# 设置一个模型，该模型返回每个目标层的激活值 (以字典方式)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

def compute_loss(input_image):
    features = feature_extractor(input_image)
    # 初始化损失
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # 通过仅在损失中涉及非边界像素，我们避免了边界伪影。
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss


@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # 计算梯度
    grads = tape.gradient(loss, img)
    # 归一化梯度
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img


original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  # 准备一个备份
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.preprocessing.image.save_img(result_prefix + ".png", deprocess_image(img.numpy()))


display(Image(result_prefix + ".png")) # 训练结果图
