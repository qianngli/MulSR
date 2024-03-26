# [MulSR](https://ieeexplore.ieee.org/document/10242161 "MulSR")  
![Image text](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/architecture.png)

## Abstract  
Hyperspectral image super-resolution (SR) using auxiliary RGB image has obtained great success. Currently, most methods, respectively, train single model to handle different scale factors, which may lead to the inconsistency of spatial and spectral contents when converted to the same size. In fact, the manner ignores the exploration of potential interdependence among different scale factors in a single model. To this end, we propose a multiscale factor joint learning for hyperspectral image SR (MulSR). Specifically, to take advantage of the inherent priors of spatial and spectral information, a deep architecture using single scale factor is designed in terms of symmetrical guided encoder (SGE) to explore the hyperspectral image and RGB image. Considering that there are obvious differences in texture details at various scale factors, another architecture is proposed which is basically the same as above, except that its scale factor is larger. On this basis, a multiscale information interaction (MII) unit is modeled between two architectures by a direction-aware spatial context aggregation (DSCA) module. Besides, the contents generated by the model with multiscale factor are combined to build a learnable feedback compensation correction (LFCC). The difference is fed back to the architecture with large scale factor, forming an interactive feedback joint optimization pattern. This calibrates the representation of spatial and spectral contents in the reconstruction process. Experiments on synthetic and real datasets demonstrate that our MulSR shows superior performance in terms of qualitative and quantitative aspects.

## Dataset  
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](https://dataverse.harvard.edu/ "Harvard"), and [Sample of Roman Colosseum](https://earth.esa.int/eogateway/missions/worldview-2 "Sample of Roman Colosseum"), are employed to verify the effectiveness of the proposed MulSR.**

## Requirement  
**PyTorch, NVIDIA GeForce GTX 1080 GPU.**

## Implementation  
**CAVE and Harvard datasets:** We select 80% samples to train. Then, these samples are randomly flipped, rotated, and rolled.  
**Sample of Roman Colosseum dataset:** The image in the training set is randomly cropped to obtain 64 patches with the size 12 × 12 β. Similarly, these patches are augmented by above way.  
  
In test stage, anisotropic Gaussian is first applied to blur the HR hyperspectral images. Then, we downsample the blur images according to scale factor and add Gaussian noise to obtain test images. Here, the mean and variance of parameters are set to 0 and 0.001, respectively. With respect to experimental setup, we select the size of convolution kernels to be 3 × 3, except for the kernels mentioned above. Moreover, the number of these kernels is set to 64. Following previous works, we fix the learning rate at 10^(−4), and its value is halved every 30 epoch. To optimize our model, the ADAM optimizer with β1 = 0.9 and β2 = 0.99 is chosen. Moreover, we set 2α = β in our article.

## Result  
To evaluate the performance, we apply peak signal-to-noise ratio (PSNR), structural similarity (SSIM), spectral angle mapper (SAM), and root mean squared error (RMSE).  
The  bold represent the way used in this article and best performance.  
- Effect of different parts for MII unit on the performance  
  | Module | PNSR | SSIM | SAM | RMSE | Param. |
  |:------:|:----:|:----:|:---:|:----:|:-----:|
  | w/o MII | 43.686 | 0.9910 | 3.904 | 0.0069 | **2427K** |
  | **w/ MII** | **44.745** | **0.9923** | **3.694** | **0.0061** | 2509K |
  | w/o DSCA | 43.487 | 0.9901 | 4.337 | 0.0071 | 2452K |
- Effect of LFCC on the performance  
  | Module | PNSR | SSIM | SAM | RMSE | Param. |
  |:------:|:----:|:----:|:---:|:----:|:-----:|
  | w/o LFCC | 44.556 | **0.9923** | 3.752 | 0.0063 | **2381K** |
  | **w/ LFCC** | **44.745** | **0.9923** | **3.694** | **0.0061** | 2509K |
- Effect of multiscale strategy on the performance  
  | Type | PNSR | SSIM | SAM | RMSE | Param. |
  |:----:|:----:|:----:|:---:|:----:|:-----:|
  | Single scale | 42.346 | 0.9866 | 5.196 | 0.0080 | **1187K** |
  | **Multi-scale** | **44.745** | **0.9923** | **3.694** | **0.0061** | 2509K |
- Effect of different numbers of cross-modal fusion module on the performance  
  | Metrics | 1 | 2 | 3 | 4 |
  |:-------:|:------:|:------:|:-----:|:------:|
  | PNSR | 44.199 | 44.745 | 44.835 | **45.438** |
  | SSIM | 0.9911 | 0.9923 | 0.9925 | **0.9932** |
  | SAM | 3.904 | 3.694 | 3.536 | **3.486** |
  | RMSE | 0.0065 | 0.0061 | 0.0061 | **0.0057** |
  | Param. | **1446K** | 2509K | 3572K | 4636K |
  
## Performance Comparison With Existing Approaches
To show the superiority of the proposed method, we compare the proposed method with four existing approaches on different scale factors and datasets, including PZRes-Net, MoG-DCN, UAL, and CoarseNet.  

- Fig. 1 shows the visual comparison of spatial reconstruction. One can observe that our method obtains more bluer in the enlarged area. In particular, the contents around the edges are very light in this area.  
  ![Fig. 1](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/Fig5.png)
- The visual comparison of spectral distortion is displayed in Fig. 2. Likewise, the red curves of our MulSR are closer to ground truth.  
  ![Fig. 2](https://raw.githubusercontent.com/qianngli/Images/master/MulSR/Fig6.png)

## References
[1] **Q. Li**, Y. Yuan, X. Jia, and Q. Wang, “Dual-stage approach toward hyperspectral image super-resolution,” *IEEE Trans. Image Process.*, vol. 31, pp. 7252–7263, 2022.  
[2] **Q. Li**, M. Gong, Y. Yuan, and Q. Wang, “RGB-induced feature modulation network for hyperspectral image super-resolution,” *IEEE Trans. Geosci. Remote Sens.*, vol. 61, 2023, Art. no. 5512611.  
[3] **Q. Li**, M. Gong, Y. Yuan, and Q. Wang, “Symmetrical feature propagation network for hyperspectral image super-resolution,” *IEEE Trans. Geosci. Remote Sens.*, vol. 60, 2022, Art. no. 5536912.  
[4] Q. Wang, **Q. Li**, and X. Li, “Hyperspectral image superresolution using spectrum and feature context,” *IEEE Trans. Ind. Electron.*, vol. 68, no. 11, pp. 11276–11285, Nov. 2021.  
[5] Q. Wang, **Q. Li**, and X. Li, “A fast neighborhood grouping method for hyperspectral band selection,” *IEEE Trans. Geosci. Remote Sens.*, vol. 59, no. 6, pp. 5028–5039, Jun. 2021.  

## Citation 
**Please consider cite this paper if you find it helpful.**

    @ARTICLE{10242161,
      author={Li, Qiang and Yuan, Yuan and Wang, Qi},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Multiscale Factor Joint Learning for Hyperspectral Image Super-Resolution}, 
      year={2023},
      volume={61},
      number={},
      pages={1-10},
      keywords={Hyperspectral imaging;Feature extraction;Superresolution;Analytical models;Spatial resolution;Fuses;Task analysis;Compensation correction;contextual aggregation;hyperspectral image;joint optimization;super-resolution (SR)},
      doi={10.1109/TGRS.2023.3312436}
    }
  
--------
  
If you has any questions, please send e-mail to liqmges@gmail.com.
