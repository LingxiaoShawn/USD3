# USD3
Official pytorch source code for 

> [Improving and Unifying Discrete-time and Continuous-time **Discrete Denoising Diffusion**](https://arxiv.org/pdf/2402.03701.pdf)   
> Lingxiao Zhao*, Xueying Ding*, Lijun Yu, Leman Akoglu

## About  
**Discrete** diffusion models are less explored comparing with continuous-state diffusion models, yet there are many dicrete data like language and graph. We have **unified and simplified** the diffusion framework (forward diffusion process, backward denoising process, and loss computation) for *discrete-time and continuous-time* denoising diffusion. Notice that the framework currently works on *nominal data* where categorical classes are NOT ordered. 

#### Supported features:
  
* ✅ **Unified code**: you only need to switch the loss function to choose between continuous time and discrete time. Forward and backward process are shared with the same code. 
* ✅ **Fast and memory efficient**: forward and backward process does NOT store the costly $C\times C$ transition matrices, thanks to the nominal data assumption. We provide both efficient exact VLB loss and simplified VLB loss. Backward process easily supports jump steps. 
* ✅ **Any dimension input**: our code easily support any multi-element object $X$ with dimensionality $(B, N_1,...,N_k, C)$ *without* any modification, where $k$ can be any positive integer. $B$ is the batch size. If samples have different number of elements, you can provide the mask of paddings to the loss function, which will ignore these padding elements. 
* ✅ **Conditional diffusion**: one can provide the mask for conditional part, and these elements won't change during the conditional diffusion process. 
* ✅ **Element-dependent noise**: we support two types of noise. 1) all elements share the same noise, with the categorical noise distribution having shape $(C)$. 2) element-dependent noise with noise distribution shape $(B, N_1,...,N_k, C)$. This is particularly useful in conditional diffusion process, where one can define element-dependent noise. 


## Installation 

### If you only use [discrete_diffusion.py](./discrete_diffusion.py) for your own project

As long as you have pytorch (>= 1.13) installed, you are free to use directly :)


### Run experiment in the paper 
We follow the experimental setup and code base from [TauLDR](https://github.com/andrew-cr/tauLDR). 



## Usage 

### If you only use [discrete_diffusion.py](./discrete_diffusion.py) for your own project


### Run experiment in the paper 



## Citation 
If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{zhao2024improving,
  title={Improving and Unifying Discrete\&Continuous-time Discrete Denoising Diffusion},
  author={Zhao, Lingxiao and Ding, Xueying and Yu, Lijun and Akoglu, Leman},
  journal={arXiv preprint arXiv:2402.03701},
  year={2024}
}
```