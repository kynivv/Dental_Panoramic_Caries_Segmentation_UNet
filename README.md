# About Dataset:

The Panoramic dental dataset is a collection of images and their corresponding dental caries segmentation masks.
Data: The data was taken from an open source on github: https://github.com/Zzz512/MLUA.

The dataset contains:
**images:** Full-size dental x-rays images;
**labels:** Full-size caries segmentation mask;
**images_cut:** Crop dental x-rays images at the border of the teeth;
**labels_cut:** Full-size caries segmentation mask at the border of the teeth;
**annotations.bboxes_caries:** Bboxes of each area of ​​​​caries in the full-size images;
**annotations.bboxes_teeth:** Bboxes of each teeth in the full-size images.


# U-NET Architecture of the model:

## Convolutional Layers:
The model starts with a convolutional layer with 64 filters, a kernel size of (3, 3), ReLU activation function, He normal initialization, and same padding.
A dropout layer with a dropout rate of 0.1 follows the first convolutional layer.
Another convolutional layer with similar configurations is added, followed by a max pooling layer with a pool size of (2, 2).

## Repeat Block:
The model repeats a similar pattern for two more blocks. Each block consists of two convolutional layers with dropout, and the last layer in each block is a max pooling layer.

## Up-sampling and Transpose Convolution:
After the contracting path, the model includes an expanding path. It starts with a convolutional layer with 256 filters, a kernel size of (3, 3), ReLU activation, He normal initialization, and same padding. This is followed by a dropout layer.
Another convolutional layer with similar configurations is added.
The model then uses a transpose convolution (sometimes called deconvolution or up-sampling) with a kernel size of (2, 2) and strides of (2, 2) to up-sample the feature maps.

## Repeat Up-sampling Block:
The up-sampling and convolution pattern is repeated with slight modifications, including different filter sizes.

## Final Convolutional Layer:
The final layers consist of a convolutional layer with 3 filters (assuming this is a segmentation task with three output channels corresponding to the classes) and a kernel size of (1, 1) with a sigmoid activation function.
