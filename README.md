# BokehDiff: Neural Lens Blur with One-Step Diffusion

![45 Teaser](banner/45-teaser.jpg)
![13 Teaser](banner/13-teaser.jpg)
![0 Teaser](banner/0-teaser.jpg)

- A physics-inspired self-attention (PISA) module design that aligns with the image formation process, incorporating depth-dependent circle of confusion constraint and self-occlusion effects. 
- A one-step inference scheme to exploit the diffusion prior, without introducing additional noise.
- A scalable paired data synthesis scheme, combining AIGC photorealistic foregrounds with transparency and conventional all-in-focus background images, balancing authenticity and scene diversity.

The dataset synthesis is now performed on-the-fly, which means it only needs to take foreground images (with transparency) and background images as input, and the images with lens blur will be generated in `dataset.py` in parallel with training.

I plan to open-source the code in the following weeks. Stay tuned for updates!