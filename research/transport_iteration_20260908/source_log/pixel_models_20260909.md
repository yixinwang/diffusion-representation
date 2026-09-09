# Primary-source comparison, September 9

Search scope: “Wavelet Flow Fast Training of High Resolution Normalizing Flows paper” and “pixel space flow matching diffusion no VAE efficient PixelFlow PixelDiT paper.” Only primary papers and author repositories support the following comparison.

[Wavelet Flow](https://arxiv.org/abs/2010.13821), by Yu, Derpanis and Brubaker, already uses a multiscale wavelet factorization with conditional normalizing flows and reports faster training than earlier normalizing flows. Our exact coupling composition shares this central construction. Learned pre-analysis and a globally attending packed residual conditioner change the parameterization; they do not establish a new factorization theorem or superiority over diffusion. The [author implementation](https://github.com/YorkUCVIL/Wavelet-Flow) is an additional reproduction source.

[PixelFlow](https://arxiv.org/abs/2504.07963), by Chen and colleagues, already models raw pixels with a cascade of flow models and removes a pretrained VAE. It reports ImageNet and text-to-image experiments. Direct pixels and a coarse-to-fine design are therefore existing mechanisms. Our proposed finite coupling maps seek explicit inversion and densities with few transformations. That distinction supplies a falsifiable numerical and computational question, without supplying a quality advantage.

[PixelDiT](https://arxiv.org/abs/2511.20645), by Yu and colleagues, separates global patch processing from pixel detail processing in a VAE-free diffusion model. Its [author repository](https://github.com/NVlabs/PixelDiT) reports released training and inference code. A packed globally attending FM is a small control related to this computational design; it is not a reproduced PixelDiT benchmark. Future high-resolution claims must compare against strong pixel-space methods as well as latent methods under matched information and budgets.

Scientific role: these papers constrain novelty and baseline strength. Their published performance is not evidence for our implementation. No external generator weights or training observations were acquired in this source check.
