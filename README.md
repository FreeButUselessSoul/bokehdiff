# BokehDiff: Neural Lens Blur with One-Step Diffusion

- BokehDiff employs a physics-inspired self-attention (PISA) module that aligns with the image formation process, incorporating depth-dependent circle of confusion constraint and self-occlusion effects. 
- BokehDiff adapts the diffusion model to the one-step inference scheme without introducing additional noise, and achieve results of high quality and fidelity.
- To address the lack of scalable paired data, we propose to synthesize photorealistic foregrounds with transparency with diffusion models, balancing authenticity and scene diversity.

The dataset synthesis is now performed on-the-fly, which means it only needs to take foreground images (with transparency) and background images as input, and the images with lens blur will be generated in `dataset.py` in parallel with training.

I plan to open-source the code in the following weeks. Stay tuned for updates!