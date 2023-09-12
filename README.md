# TransMorph with Deformable Cross-attention (DCA)
[PyTorch] Deformable Cross-Attention Transformer for Medical Image Registration\
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2303.06179-b31b1b.svg)](https://arxiv.org/abs/2303.06179)

This is a **PyTorch** implementation of my paper:\
**Accepted to MLMI 2023!**\
<a href="https://arxiv.org/abs/2303.06179">Chen, Junyu, et al. "Deformable Cross-Attention Transformer for Medical Image Registration." In Machine Learning in Medical Imaging: 14th International Workshop, MLMI 2023.</a>

Pretrained weights on the OASIS dataset from Learn2Reg challenge: [TransMorph-DCA](https://drive.google.com/uc?export=download&id=1QfMiTzZMIlBDg8nI9NIWf0FyRHc2ZNXo)\
Pretrained weights on the [IXI dataset](https://github.com/junyuchen245/Preprocessed_IXI_Dataset): [TransMorph-DCA](https://drive.google.com/uc?export=download&id=1_3qh9pqHwikUUtbDy-ANvcnr532u8K-1)
## Deformable cross-attention illustration:
<img src="https://github.com/junyuchen245/TransMorph_DCA/blob/main/figs/cross_attn.jpg" width="600"/>

## TransMorph-DCA Architecture:
<img src="https://github.com/junyuchen245/TransMorph_DCA/blob/main/figs/DefTransMorph.jpg" width="800"/>

## Example Results:
<img src="https://github.com/junyuchen245/TransMorph_DCA/blob/main/figs/GradCamCompare.jpg" width="1000"/>


## Citation:
If you find this code is useful in your research, please consider to cite:
    
    @article{chen2023deformable,
    title={Deformable Cross-Attention Transformer for Medical Image Registration},
    author={Chen, Junyu and Liu, Yihao and He, Yufan and Du, Yong},
    journal={arXiv preprint arXiv:2303.06179},
    year={2023}
    }
    
