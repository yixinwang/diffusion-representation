# Image/video baseline map

Primary sources checked on 23 August 2026:

- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752): canonical autoencoder-plus-latent-diffusion baseline.
- Blattmann et al., [Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf): latent video diffusion with fixed image encoding and temporal layers.
- Yu et al., [Video Probabilistic Diffusion Models in Projected Latent Space](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.html): projected video latents motivated by high-dimensional compute and memory.
- Ni et al., [Conditional Image-to-Video Generation with Latent Flow Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.html): a motion-focused latent diffusion baseline for conditional video.
- Ma et al., [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048): transformer processing of spatiotemporal latent tokens.
- Gupta et al., [Photorealistic Video Generation with Diffusion Models (W.A.L.T)](https://arxiv.org/abs/2312.06662): latent video diffusion transformer and super-resolution cascade.
- Phung et al., [Wavelet Diffusion Models Are Fast and Scalable Image Generators](https://openaccess.thecvf.com/content/CVPR2023/html/Phung_Wavelet_Diffusion_Models_Are_Fast_and_Scalable_Image_Generators_CVPR_2023_paper.html): wavelet-domain diffusion; mandatory prior-art and image baseline check.
- Lu et al., [Multi-Resolution Continuous Normalizing Flows](https://arxiv.org/abs/2106.08462): WaveletFlow, an invertible multiresolution flow; mandatory prior-art and likelihood baseline check.
- Meng et al., [On Distillation of Guided Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/html/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.html): few-step distilled diffusion; prevents comparison only to obsolete long-step samplers.

These papers establish relevant baseline families, not that the proposed method improves on them. Any observed-data claim requires runnable, compute-matched implementations and cannot substitute published scores across different preprocessing or evaluator settings.
