"""gan_data_processing — preprocessing, metrics, and losses for GAN-based MRI synthesis.

Subpackages:
    - preprocessing: nnUNet-style I/O, cropping, resampling, normalization.
    - metrics:       MAE/MSE/NMSE/PSNR/SSIM and enhancing-tumor metrics.
    - losses:        tumor-aware and ET-aware GAN losses.

Import from the relevant subpackage rather than the top level, e.g.:
    from gan_data_processing.metrics import ssim, masked_ssim
    from gan_data_processing.losses import TumorAwareGANLoss
    from gan_data_processing.preprocessing import preprocess_dataset
"""

__version__ = "0.1.0"
