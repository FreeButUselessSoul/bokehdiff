# BokehDiff: Neural Lens Blur with One-Step Diffusion

- A physics-inspired self-attention (PISA) module design that aligns with the image formation process, incorporating depth-dependent circle of confusion constraint and self-occlusion effects. 
- A one-step inference scheme to exploit the diffusion prior, without introducing additional noise.
- To address the lack of scalable paired data, we propose to synthesize photorealistic foregrounds with transparency with diffusion models, balancing authenticity and scene diversity.

The dataset synthesis is now performed on-the-fly, which means it only needs to take foreground images (with transparency) and background images as input, and the images with lens blur will be generated in `dataset.py` in parallel with training.

I plan to open-source the code in the following weeks. Stay tuned for updates!