# FAPEloss

### AlphaFold Algorithm
![image](https://tvax1.sinaimg.cn/large/005vYU31ly1gu0alzx2yhj618c0h87d602.jpg)

### Requirements
😄 To install requirements:
```
torch
einops
```
### Usage
1. First step
```
$git clone https://github.com/wangleiofficial/FAPEloss.git
```

2. Testing FAPE Loss
```
from fape import FAPEloss
import torch

# define the transformation
predict_T = (torch.randn((1, 1, 3, 3)), torch.randn((1, 1, 3)))
transformation = (torch.randn((1, 1, 3, 3)), torch.randn((1, 1, 3)))

# define loss
fape = FAPEloss()
loss = fape(predict_T, transformation)
```

### Citing AlphaFold paper
```
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  doi     = {10.1038/s41586-021-03819-2},
  note    = {(Accelerated article preview)},
}
```
