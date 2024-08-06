# USD3
Official pytorch source code for 

> [Improving and Unifying Discrete-time and Continuous-time **Discrete Denoising Diffusion**](https://arxiv.org/pdf/2402.03701.pdf)   
> Lingxiao Zhao*, Xueying Ding*, Lijun Yu, Leman Akoglu

## Update: simpler and faster continuous-time loss

We have discovered that the continuous loss we derived can be further simplified without any approximations, resulting in a much faster computation that requires only a single forward pass of the model. The code has already been updated, and we will soon update the arXiv paper to include the derivation. Note that the simplified continuous loss is similar to the loss described in [SEDD](https://arxiv.org/pdf/2310.16834), though our derivation stems from a completely different perspective: we derive it directly from the VLB bound shown in [CTMC](https://arxiv.org/abs/2205.14987). This indicates that concrete socre matching is the same as VLB derivation from CTMC, which is also shown in continuous-state model (score matching is the same as VLB).  


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

* **Train**
1. Create `UnifiedDiscreteDiffusion` object

    ``` python 
    from discrete_diffusion import UnifiedDiscreteDiffusion
    diffusion = UnifiedDiscreteDiffusion(num_steps, # 0 means use continuous time
                                         num_classes, 
                                         noise_schedule_type, 
                                         noise_schedule_args)
    ```

2. Sampling `x_t` from `t` and `x_0` (every batch)
      * For continuous-time case (`num_steps=0`),  `t` should in the range  0 ~ 1.0
      * For discrete-time, `t` should in integer in the range  `0 ~ num_steps`
      * `m` is the noise distribution, see code for doc
      * `conditional_mask` is used for keeping certain part unchanged or conditioned
    ``` python
    x_t = diffusion.qt_0_sample(x_0, t, m, conditional_mask)
    ```

3. Compute loss with input the noisy `x_t` and original `x_0` (every batch)
    * Assume you have a `model` (network): (B, N1, ..., Nk), t -> (B, N1, ..., Nk, C), where C is `num_classes`
    * `model` takes `x_t` and `t` as input, and output prediction of `x_0` distribution
    ``` python
    logits_t = model(x_t, t)

    # loss = coeff_ce * ce + coeff_vlb * vlb
    loss = diffusion.compute_loss(logits_t,
                                  x_t, 
                                  x_0, 
                                  t, 
                                  m, 
                                  coeff_ce=0.1,
                                  coeff_vlb=1.0, 
                                  conditional_mask=conditional_mask,
                                  simplified_vlb=False)
    ```
    * There are three parameters to play (`coeff_ce`, `coeff_vlb`, `simplified_vlb`), see paper for detail.

4. Update model with 
```loss['loss'].backward()``` (every batch)

* **Generation**  
  * After training, you can use the trained `model` to generate samples. 
  * In discrete-time case, one would want `num_backward_steps` to be smaller than the training steps `num_steps` for good performance.  

  ``` python
  diffusion.sample(model,
                   num_backward_steps, 
                   m, 
                   conditional_mask=None,
                   conditional_input=None)
  ```
  * One can also use mcmc refinement in sampling, see code doc for parameters. 


### Run experiment in the paper 


## TODO
- [ ] Add "Run experiment in the paper"
  - [ ] Installation
  - [ ] Usage
  - [ ] Code


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