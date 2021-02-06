# Keras  

## [Neural style transfer](./neural_style_transfer.py)  
> Author: [fchollet](https://twitter.com/fchollet)  
> Description: Transfering the style of a reference image to target image using gradient descent.  

### Introduction  
Style transfer consists in generating an image with the same "content" as a base image, but with the "style" of a different picture (typically artistic).  
This is achieved through the optimization of a loss function that has 3 components: "style loss", "content loss", and "total variation loss":  
- The total variation loss imposes local spatial continuity between the pixels of the combination image, giving it visual coherence.  
- The style loss is where the deep learning keeps in --that one is defined using a deep convolutional neural network. Precisely, it consists in a sum of L2 distances between the Gram matrices of the representations of the base image and the style reference image, extracted from different layers of a convnet (trained on ImageNet). The general idea is to capture color/texture information at different spatial scales (fairly large scales --defined by the depth of the layer considered).  
- The content loss is a L2 distance between the features of the base image (extracted from a deep layer) and the features of the combination image, keeping the generated image close enough to the original one.  


## [deep dream](./deep_dream.py)  
> Author: [fchollet](https://twitter.com/fchollet)  
> Description: Generating Deep Dreams with Keras.  

### Introduction  
"Deep dream" is an image-filtering technique which consists of taking an image classification model, and running gradient ascent over an input image to try to maximize the activations of specific layers (and sometimes, specific units in specific layers) for this input. It produces hallucination-like visuals.  
It was first introduced by Alexander Mordvintsev from Google in July 2015.  

Process:  
- Load the original image.  
- Define a number of processing scales ("octaves"), from smallest to largest.  
- Resize the original image to the smallest scale.  
- For every scale, starting with the smallest (i.e. current one): - Run gradient ascent - Upscale image to the next scale - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size. To obtain the detail lost during upscaling, we simply take the original image, shrink it down, upscale it, and compare the result to the (resized) original image.  

### Problems that may arise  
File "D:\Anaconda3\envs\TF2.1\lib\site-packages\tensorflow_core\python\keras\saving\hdf5_format.py", line 651, in load_weights_from_hdf5_group  
    original_keras_version = f.attrs['keras_version'].decode('utf8')  
AttributeError: 'str' object has no attribute 'decode'  

### Solution  
Uninstall the original h5py module and install version 2.10  
> pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/  

