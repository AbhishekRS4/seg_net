# SegNet implementation on Cityscapes dataset

## Notes
* Implementation of SegNet variant with VGG-16
* The main change is in the upsampling mechanism that is used in the original model
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Unpooling mechanism used to upsample features which uses the indices from pooling stage.

## To do
- [x] convolution transpose upsampling
- [x] bilinear upsampling
- [x] nearest neighbor upsampling
- [x] bayesian with convolution transpose upsampling
- [ ] Compute metrics
- [ ] Sample output

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [Seg\_Net](https://arxiv.org/pdf/1511.00561.pdf)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
