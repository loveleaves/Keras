import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
"""
如果想训练自己的图片，上面代码改成如下代码：
base_image_path=“your_path.jpg” # base_image_path为待风格迁移的图片地址
style_reference_image_path="your_path.jpg" # style_reference_image_path 为风格样式图片地址
"""

result_prefix = "paris_generated"

# 各部分损失的权重设置
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# 生成图片的尺寸
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

#通过下面命令查看要进行风格迁移的基本图片和样式参考图片
from IPython.display import Image, display

display(Image(base_image_path))
display(Image(style_reference_image_path))

# 图像预处理
def preprocess_image(image_path):
    # 利用Keras库函数的来打开图片，调整图片大小并将其格式化为适当的张量
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # 再利用函数将张量转换为有效图像
    x = x.reshape((img_nrows, img_ncols, 3))
    # 通过平均像素去除零中心
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# 图像张量的gram矩阵（特征矩阵和特征矩阵转置的乘积）

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

# “风格损失”旨在保持生成图像中参考图像的样式。
# 它基于的gram矩阵（样式提取）来自样式参考图像
# 和从它生成的图像的特征图

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# 辅助损失函数设计来是为了
# 维护生成的图像中的基本图像的内容

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

# 第三个损失函数是总变化损失，
# 设计此函数是为了使生成的图像保持局部连贯。

def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


"""
接下来，让我们创建一个特征提取模型，该模型检索VGG19的中间激活（根据名字制成字典）。
"""

# 建立一个加载了已经训练好的ImageNet的权重的VGG19模型
model = vgg19.VGG19(weights="imagenet", include_top=False)

# 获取每个“关键”层的符号输出（我们给它们指定了唯一的名称）。
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# 建立一个模型，以返回VGG19中每层的激活值（以字典的方式）。
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

"""
最后，这是计算样式转移损失的代码。
"""

# 用于样式丢失的图层列表。
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# 用于内容丢失的层。
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # 初始化损失
    loss = tf.zeros(shape=())

    # 加入内容丢失
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # 加入风格损失
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # 加入总变化损失
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


"""
将tf.function装饰器添加到损耗计算和梯度计算中
使在编译过程中能运行更快
"""

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

"""
重复执行批量梯度下降步骤，以最大程度地减少损失，并每100次迭代保存生成的图像。
每100步将学习率降低0.96。
"""

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, img)

"""
经过4000次迭代，输出结果：
"""

display(Image(result_prefix + "_at_iteration_4000.png"))
