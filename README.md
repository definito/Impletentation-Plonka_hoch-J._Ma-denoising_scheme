Implementation of plonka hoch algorithm

jupyter nb file ` implementation-tesing_methods & Theory.ipynb` contains all the testing of methods and theorem proof. etc.

# The main python file- `plonka_hoch_denoising.py` however contains the main implementation

# Plonka-Hoch Denoising Method

This Python script contains a Python implementation of the Plonka-Hoch Denoising Method, which was proposed in the paper "CONVERGENCE OF AN ITERATIVE NONLINEAR SCHEME FOR DENOISING OF PIECEWISE CONSTANT IMAGES".

This denoising method is used to remove noise from images while preserving the sharpness of their edges.

## Installation

This project requires Python 3.7 or later.

You will also need to install the following Python packages:

* numpy
* cv2
* matplotlib
* PIL
* queue
* os

Note: You don't need to install the `queue` and `os` packages as they are part of the Python standard library.

## How to use

1. Clone the repository:

```bash
  git@gitlab.gwdg.de:debwashis.borman/Plonka_hoch-Jianwei_Ma-denoising_scheme.git
```

Import the `PlonkaDenoiseMasterClass` and `PlonkaHochDenoising` classes from the script. The `PlonkaHochDenoising` class is a subclass of the `PlonkaDenoiseMasterClass` class and contains additional methods.

Instantiate an instance of `PlonkaHochDenoising` with the necessary parameters, which include:

*`image_path`: The path to the image to be denoised.

*`resize_shape`: The desired size of the image. Default is (50, 50).

*`sigma`: The standard deviation of the noise added to the image. Default is 10.

*`theta`: The threshold for denoising. Default is 15.

*`alpha`: The weight parameter in the iteration scheme. Default is 0.1.

*`num_iter`: The number of iterations to run the denoising algorithm. Default is 10.

*`flag`: If `True`, the faster version of the iteration function will be used. Default is `False`.

After instantiation, call the `plonka_method()` method on the instance with `shrinkage_param` as the parameter, which is the threshold parameter for shrinkage function. The method will plot the original image, the noisy image, the image after iterations, the image after mean filtering, and the image after median filtering.

## Example

Here is an example of how to use the script:

<pre><codeclass="!whitespace-pre hljs language-python">import plonka_hoch_denoising as pldenoise

import numpy as np
img_path = "example-1.png"
resize_shape = (200,200)
sigma = 20
theta =50
alpha = 0.10
num_iter = 1
flag = True 


instance = pldenoise.PlonkaHochDenoising(img_path, resize_shape, sigma, theta, alpha, num_iter, flag)
instance.plonka_method(shrinkage_param=10)


</code></div></div></pre>

This will run the Plonka-Hoch denoising method on the specified image, using the specified parameters.
