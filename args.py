# Argument parameters file

import argparse

def parse_args():
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser = argparse.ArgumentParser(description='Parameter settings for end-to-end optimization of neural nano-optics')

    # Data loading arguments
    parser.add_argument('--train_dir'  , type=str, default='./training/data/train/', help='Directory of training input images')
    # parser.add_argument('--train_dir'  , type=str, required=True, help='Directory of training input images')
    parser.add_argument('--test_dir'   , type=str, default='./training/data/test/', help='Directory of testing input images')

    # Saving and logging arguments
    parser.add_argument('--save_dir'   , type=str, default='./training/save/', help='Directory for saving ckpts and TensorBoard file')
    parser.add_argument('--save_freq'  , type=int, default=50, help='Interval to save model')
    parser.add_argument('--log_freq'   , type=int, default=20, help='Interval to write to TensorBoard')
    parser.add_argument('--ckpt_dir'   , type=none_or_str, default='./training/ckpt/', help='Restoring from a checkpoint')
    parser.add_argument('--max_to_keep', type=int, default=2, help='Number of checkpoints to save')

    # Loss arguments
    parser.add_argument('--loss_mode'          , type=str, default='L1')
    parser.add_argument('--batch_weights'      , type=str, default='1.0')
    parser.add_argument('--Norm_loss_weight'   , type=float, default=1.0)
    parser.add_argument('--P_loss_weight'      , type=float, default=0.1)
    parser.add_argument('--Spatial_loss_weight', type=float, default=0.0)
    parser.add_argument('--vgg_layers'         , type=str, default='block2_conv2,block3_conv2')

    # Training arguments
    parser.add_argument('--steps'     , type=int, default=1000, help='Total number of optimization cycles')
    parser.add_argument('--aug_rotate', type=str2bool, default=False, help='True to rotate PSF during training')

    # Convolution arguments
    parser.add_argument('--real_psf'     , type=str,default='./training/data/psf.npy', help='Npy of experimentally measured PSF')
    parser.add_argument('--psf_mode'     , type=str, default='REAL_PSF', help='Use simulated PSF or captured PSF')
    parser.add_argument('--conv_mode'    , type=str, default='SIM', help='True to apply convolution for forward model')
    parser.add_argument('--conv'         , type=str, default='patch_size', help='patch_size for memory efficiency, full_size for full image')
    parser.add_argument('--do_taper'     , type=str2bool, default=False, help='Activate edge tapering')
    parser.add_argument('--offset'       , type=str2bool, default=True, help='True to use offset convolution mode')
    parser.add_argument('--normalize_psf', type=str2bool, default=False, help='True to normalize PSF')

    # Sensor arguments
    parser.add_argument('--a_poisson', type=float, default=0.00004, help='Poisson noise component')
    parser.add_argument('--b_sqrt'   , type=float, default=0.00001, help='Gaussian noise standard deviation')
    parser.add_argument('--mag'      , type=float, default=8.1, help='Relay system magnification factor (slightly less than 10x)')

    # Optimization arguments
    parser.add_argument('--Phase_iters', type=int, default=0, help='Number of meta-optic optimization iterations per cycle')
    parser.add_argument('--Phase_lr'   , type=float, default=5e-3, help='Meta-optic learning rate')
    parser.add_argument('--Phase_beta1', type=float, default=0.9, help='Meta-optic beta1 term for Adam optimizer')
    parser.add_argument('--G_iters'    , type=int, default=1, help='Number of deconvolution optimization iterations per cycle')
    parser.add_argument('--G_lr'       , type=float, default=1e-4, help='Deconvolution learning rate')
    parser.add_argument('--G_beta1'    , type=float, default=0.9, help='Deconvolution beta1 term for Adam optimizer')
    parser.add_argument('--G_network'  , type=str, default='FP', help='Select deconvolution method')
    parser.add_argument('--snr_opt'    , type=str2bool, default=False, help='True to optimize SNR parameter')
    parser.add_argument('--snr_init'   , type=float, default=3.0, help='Initial value of SNR parameter')

    args = parser.parse_args()
    # args.theta_base = [float(w) for w in args.theta_base.split(',')]
    args.batch_weights = [float(w) for w in args.batch_weights.split(',')]
    print(args)
    return args
