<div align="justify">
  <div align="center">
    
  # [Multiscale Factor Joint Learning for Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/document/10242161 "Multiscale Factor Joint Learning for Hyperspectral Image Super-Resolution")  
 
  </div>

## Abstract  
![Image text](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/architecture.png)  
Hyperspectral image super-resolution (SR) using auxiliary RGB image has obtained great success. Currently, most methods, respectively, train single model to handle different scale factors, which may lead to the inconsistency of spatial and spectral contents when converted to the same size. In fact, the manner ignores the exploration of potential interdependence among different scale factors in a single model. To this end, we propose a multiscale factor joint learning for hyperspectral image SR (MulSR). Specifically, to take advantage of the inherent priors of spatial and spectral information, a deep architecture using single scale factor is designed in terms of symmetrical guided encoder (SGE) to explore the hyperspectral image and RGB image. Considering that there are obvious differences in texture details at various scale factors, another architecture is proposed which is basically the same as above, except that its scale factor is larger. On this basis, a multiscale information interaction (MII) unit is modeled between two architectures by a direction-aware spatial context aggregation (DSCA) module. Besides, the contents generated by the model with multiscale factor are combined to build a learnable feedback compensation correction (LFCC). The difference is fed back to the architecture with large scale factor, forming an interactive feedback joint optimization pattern. This calibrates the representation of spatial and spectral contents in the reconstruction process. Experiments on synthetic and real datasets demonstrate that our MulSR shows superior performance in terms of qualitative and quantitative aspects.  

## Motivation  
As for the SR task, current hyperspectral image SR methods usually construct corresponding models according to different scale factors, respectively. Without loss of generality, the super-resolved images with different scale factors contain obvious texture differences. Additionally, the greater the scale factor is, the more serious the detail loss in the reconstructed images is. If this texture difference can be used effectively, it really enhances performance in scenarios. For example, the features with low detail loss can be utilized to guide the model learning with high detail loss. Nevertheless, the above hyperspectral image SR approaches do not build models from this perspective. Moreover, SR images obtained under different scale factors are converted to the images with same size. Ideally, the results should be consistent in terms of spatial and spectral contents. This can actually constrain the generated SR images by this manner. At present, existing hyperspectral image SR methods do not consider the consistency of information representation. Theoretically, the texture details of the image can be refined by adding more scale factors for inconsistent analysis. Although current natural SR methods utilize some scale factors to build models, they do not fully explore the interdependence by more branches. Therefore, how to utilize the model with small scale factor to assist the model with large scale factor for joint SR, so as to establish the interdependence among different scale factors in single model, needs further study.

## Dataset  
Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](https://dataverse.harvard.edu/ "Harvard"), and [Sample of Roman Colosseum](https://earth.esa.int/eogateway/missions/worldview-2 "Sample of Roman Colosseum"), are employed to verify the effectiveness of the proposed MulSR.  
**CAVE and Harvard datasets:** We select 80% samples to train. Then, these samples are randomly flipped, rotated, and rolled.  
**Sample of Roman Colosseum dataset:** The image in the training set is randomly cropped to obtain 64 patches with the size 12 × 12 β. Similarly, these patches are augmented by above way.  

## Dependencies  
**PyTorch, NVIDIA GeForce GTX 1080 GPU.**
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`

## Implementation  
In test stage, anisotropic Gaussian is first applied to blur the HR hyperspectral images. Then, we downsample the blur images according to scale factor and add Gaussian noise to obtain test images. Here, the mean and variance of parameters are set to 0 and 0.001, respectively. With respect to experimental setup, we select the size of convolution kernels to be 3 × 3, except for the kernels mentioned above. Moreover, the number of these kernels is set to 64. Following previous works, we fix the learning rate at 10^(−4), and its value is halved every 30 epoch. To optimize our model, the ADAM optimizer with β1 = 0.9 and β2 = 0.99 is chosen. Moreover, we set 2α = β in our article.

## Result  
To evaluate the performance and demonstrate the superiority of the proposed method, we apply peak signal-to-noise ratio (PSNR), structural similarity (SSIM), spectral angle mapper (SAM), and root mean squared error (RMSE) for comparison with four existing approaches—PZRes-Net, MoG-DCN, UAL, and CoarseNet—across different scale factors and datasets. The best result and second result are denoted as the bold and underline, respectively.  
![TABLE_V-VI](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/TABLE_V-VI.png)  
- Fig. 1 shows the visual comparison of spatial reconstruction. One can observe that our method obtains more bluer in the enlarged area. In particular, the contents around the edges are very light in this area.  
  ![Fig5](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/Fig5.png)
- The visual comparison of spectral distortion is displayed in Fig. 2. Likewise, the red curves of our MulSR are closer to ground truth.  
  ![Fig6](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/Fig6.png)

## Citation 
[1] **Q. Li**, Y. Yuan, X. Jia, and Q. Wang, “Dual-stage approach toward hyperspectral image super-resolution,” *IEEE Trans. Image Process.*, vol. 31, pp. 7252–7263, 2022.  
[2] **Q. Li**, M. Gong, Y. Yuan, and Q. Wang, “RGB-induced feature modulation network for hyperspectral image super-resolution,” *IEEE Trans. Geosci. Remote Sens.*, vol. 61, 2023.  
[3] **Q. Li**, M. Gong, Y. Yuan, and Q. Wang, “Symmetrical feature propagation network for hyperspectral image super-resolution,” *IEEE Trans. Geosci. Remote Sens.*, vol. 60, 2022.  
[4] Q. Wang, **Q. Li**, and X. Li, “Hyperspectral image superresolution using spectrum and feature context,” *IEEE Trans. Ind. Electron.*, vol. 68, no. 11, pp. 11276–11285, Nov. 2021.  
[5] Q. Wang, **Q. Li**, and X. Li, “A fast neighborhood grouping method for hyperspectral band selection,” *IEEE Trans. Geosci. Remote Sens.*, vol. 59, no. 6, pp. 5028–5039, Jun. 2021.  

--------
If you has any questions, please send e-mail to liqmges@gmail.com.

</div>
