# SegNet implementation on Cityscapes dataset

## Notes
* Implementation of SegNet variant with VGG-16
* The experiment is to use different upsampling techniques. The original implementation used Max-Unpooling.
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Unpooling mechanism used to upsample features. This uses the stored indices from corresponding pooling stage.

## Intructions to run
> To run training use - **python3 seg\_net\_train.py -h**
>
> To run inference use - **python3 seg\_net\_infer.py -h**
>
> This lists all possible commandline arguments

## Reference
* [VGG](https://arxiv.org/abs/1409.1556)
* [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
* [Bayesian SegNet](https://arxiv.org/pdf/1511.02680.pdf)
* [SegNet Project](http://mi.eng.cam.ac.uk/projects/segnet/)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## To do
- [x] convolution transpose upsampling
- [x] bilinear upsampling
- [x] nearest neighbor upsampling
- [x] bayesian with convolution transpose upsampling
- [ ] Visualize results
- [ ] Compute metrics

