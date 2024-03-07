[MulSR](https://ieeexplore.ieee.org/document/10242161 "MulSR")
======
**This is an implementation of Multi-Scale Factor Joint Learning for Hyperspectral Image Super-Resolution**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](https://dataverse.harvard.edu/ "Harvard"), and [Sample of Roman Colosseum](https://earth.esa.int/eogateway/missions/worldview-2 "Sample of Roman Colosseum"), are employed to verify the effectiveness of the proposed MulSR.**

Requirement
---------
**PyThon _._._, PyTorch _._._, NVIDIA GeForce GTX 1080 GPU.**

Implementation
--------
**CAVE and Harvard datasets:** We select 80% samples to train. Then, these samples are randomly flipped, rotated, and rolled.  
**Sample of Roman Colosseum dataset:** The image in the training set is randomly cropped to obtain 64 patches with the size 12 × 12 β. Similarly, these patches are augmented by above way.  

In test stage, anisotropic Gaussian is first applied to blur the HR hyperspectral images. Then, we downsample the blur images according to scale factor and add Gaussian noise to obtain test images. Here, the mean and variance of parameters are set to 0 and 0.001, respectively. With respect to experimental setup, we select the size of convolution kernels to be 3 × 3, except for the kernels mentioned above. Moreover, the number of these kernels is set to 64. Following previous works, we fix the learning rate at 10−4, and its value is halved every 30 epoch. To optimize our model, the ADAM optimizer with β1 = 0.9 and β2 = 0.99 is chosen. Moreover, we set 2α = β in our article.

Result
--------

Recommended Bands Comparison:
---------

Classification Performance Comparison:
----------

Computational Time Comparison
-------


Citation 
--------
**Please consider cite this paper if you find it helpful.**

	
	
--------
If you has any questions, please send e-mail to liqmges@gmail.com.
