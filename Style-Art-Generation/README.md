# Style Transfer
The code above is a respose to [this videos](https://www.youtube.com/watch?v=Oex0eWoU7AQ) coding challenge. The aim is to transfer the style of 2 images(style images) to a single base image(content image). The code makes use of transfer learning, using the pre-existing VGG-16 network architecture to compare image features, for both, style and content. I have tried to explain in brief below, what exactly is going on :)

## Dependencies
* Tensorflow
* Numpy
* tqdm
* Keras
* scipy
* pillow

use the ```pip install <package>``` command to install the dependencies.

## Result
![styletransfer](https://user-images.githubusercontent.com/34591573/34319656-ab4ca7ae-e80d-11e7-8c39-a720610cac48.png)
These results turned out to be pretty nice!!

# Simplified Approach
## Training
* Firstly, the two style images and the content images are converted into an array(also their dimensions are expanded to allow for the storage in a single data structure).
* These images are then normalized and converted from RBG channel to BGR channel( because thats what the VGG takes in, as the input).
* We create a placeholder for our final combination image.
* All the processed images are then stored in an input tensor which would be later passed into our model(after it has been initialized).
* Then we initialize the VGG-16 model without the topmost level(as we only want feature representations of the image and NOT the classification).
* We pass the images into the model to calculate the loss between the content image and combination image(content loss) and also between the style images and the combination image(style loss).
* The gradients are calculated and combination image is updated in order to decrease the total loss.
## Losses
* The content loss is calculated by simply measuring the euclidean distance between the feature representation of the content image and the combination image(from a layer of our choice).
* The style loss is calculated by first obtaining the gram matrix(encodes style) of feature representation of the style images and combination images at every layer of the VGG-16, and then, the squared-sum of the differences of gram matrices at each layer gives the total style loss.
* Also the variational loss is also used for denoising our final image(regularization).
* The sum of all these losses makes up the total loss which we aim to minimize

## Basic Usage
For Running, type in terminal
```
styleTransfer.py
```

# Credits
Thanks to [hnarayanan](https://github.com/hnarayanan/artistic-style-transfer) and [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) for the starter code.



