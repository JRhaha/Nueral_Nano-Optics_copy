import tensorflow as tf
import numpy as np
from networks.select import select_G
from dataset import train_dataset_sim, test_dataset_sim
from loss import G_loss
from args import parse_args

import metasurface.solver as solver
import metasurface.conv as conv
import scipy.optimize as scp_opt

import os
import time

## Logging for TensorBoard
def log(img, gt_img, G, snr, vgg_model, summary_writer, step, params, args):
    # Metasurface simulation
    if args.psf_mode == 'REAL_PSF':
        real_psf = np.load(args.real_psf)
        # real_psf = real_psf[0:1, :, :, :]
        real_psf = tf.constant(real_psf, dtype=tf.float32)
        real_psf = tf.image.resize_with_crop_or_pad(real_psf, params['psf_width'], params['psf_width'])
        real_psf = real_psf / tf.reduce_sum(real_psf, axis=(1,2), keepdims=True)
        psfs_debug        = real_psf
        psfs_conv_forward = real_psf
        psfs_conv_deconv  = real_psf
    else:
        assert False, ("Unsupported PSF mode")

    conv_image = params['conv_fn'](img, psfs_conv_forward)
    sensor_img = solver.sensor_noise(conv_image, params)
    _, G_img, G_debug = params['deconv_fn'](sensor_img, psfs_conv_deconv, snr, G, training=False)#G_IMG是(152,152),应该是(154,154)
    G_img = tf.image.resize_with_crop_or_pad(G_img, params['out_width'], params['out_width'])
    # Losses
    gt_img = tf.image.resize_with_crop_or_pad(gt_img, params['out_width'], params['out_width'])
    G_Content_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)

    # Save records to TensorBoard
    with summary_writer.as_default():
        # Images
        tf.summary.image(name = 'Input/Input' , data=img, step=step)
        tf.summary.image(name = 'Input/GT' , data=gt_img, step=step)

        if args.offset:
            num_patches = 3 - 1
        else:
            num_patches = 3
        for i in range(num_patches):
            tf.summary.image(name = 'Output/Output_'+str(i), data=G_img[i:i+1,:,:,:], step=step)
            tf.summary.image(name = 'Blur/Blur_'+str(i), data=conv_image[i:i+1,:,:,:], step=step)
            tf.summary.image(name = 'Sensor/Sensor_'+str(i), data=sensor_img[i:i+1,:,:,:], step=step)
            for j, debug in enumerate(G_debug):
                tf.summary.image(name = 'Debug/Debug_'+str(j)+'_'+str(i), data=debug[i:i+1,:,:,:] , step=step)

        # PSF
        for i in range(np.size(3)):#视场个数
            psf_patch = psfs_debug[i:i+1,:,:,:]
            tf.summary.image(name='PSF/PSF_'+str(i),
                                data=psf_patch / tf.reduce_max(psf_patch), step=step)
            for l in range(3):#波长个数
                psf_patch = psfs_debug[i:i+1,:,:,l:l+1]
                tf.summary.image(name='PSF_'+str("视场1")+'/PSF_'+str(i),
                                data=psf_patch / tf.reduce_max(psf_patch), step=step)

        # Metrics
        tf.summary.scalar(name = 'metrics/G_PSNR', data = G_metrics['PSNR'], step=step)
        tf.summary.scalar(name = 'metrics/G_SSIM', data = G_metrics['SSIM'], step=step)
        tf.summary.scalar(name = 'snr', data = snr, step=step)

        # Content losses
        tf.summary.scalar(name = 'loss/G_Content_loss', data = G_Content_loss_val, step=step)
        tf.summary.scalar(name = 'loss/G_Norm_loss'   , data = G_loss_components['Norm'], step=step)
        tf.summary.scalar(name = 'loss/G_P_loss'      , data = G_loss_components['P'], step=step)
        tf.summary.scalar(name = 'loss/G_Spatial_loss', data = G_loss_components['Spatial'], step=step)


## Optimization Step
def train_step(mode, img, gt_img, G, G_optimizer, snr, vgg_model, params, args):
    with tf.GradientTape() as G_tape:
        # Metasurface simulation

        if args.psf_mode == 'REAL_PSF':
            real_psf = np.load(args.real_psf)
            real_psf = real_psf[0:1, :, :, :]
            real_psf = tf.constant(real_psf, dtype=tf.float32)
            real_psf = tf.image.resize_with_crop_or_pad(real_psf, params['psf_width'], params['psf_width'])
            real_psf = real_psf / tf.reduce_sum(real_psf, axis=(1,2), keepdims=True)
            psfs_debug        = real_psf
            psfs_conv_forward = real_psf
            psfs_conv_deconv  = real_psf
        else:
            assert False, ("Unsupported PSF mode")

        conv_image = params['conv_fn'](img, psfs_conv_forward)
        sensor_img = solver.sensor_noise(conv_image, params)
        _, G_img, _ = params['deconv_fn'](sensor_img, psfs_conv_deconv, snr, G, training=True)
        G_img = tf.image.resize_with_crop_or_pad(G_img, params['out_width'], params['out_width'])

        # Losses
        gt_img = tf.image.resize_with_crop_or_pad(gt_img, params['out_width'], params['out_width'])
        G_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)

    # Apply gradients
    if mode == 'G':
        G_vars = G.trainable_variables
        if args.snr_opt:
            G_vars.append(snr)
        G_gradients = G_tape.gradient(G_loss_val, G_vars)
        G_optimizer.apply_gradients(zip(G_gradients, G_vars))
        if args.snr_opt:
            snr.assign(tf.clip_by_value(snr, 3.0, 4.0))
    else:
        assert False, "Non-existant training mode"
    
## Training loop
def train(args):
    ## Metasurface
    params = solver.initialize_params(args)
    print('Image width: {}'.format(params['image_width']), flush=True)

    # SNR term for deconvolution algorithm Winner滤波时SNR是优化变量
    snr = tf.Variable(args.snr_init, dtype=tf.float32)

    # Do not optimize phase during finetuning 此时用的是已制造好的镜片，所以将Phase的迭代次数设为0，只需要微调反卷积的参数
    if args.psf_mode == 'REAL_PSF':
        assert(args.Phase_iters == 0)

    params['conv_fn'] = conv.convolution_tf(params, args)#将输入图像与PSF卷积
    params['deconv_fn'] = conv.deconvolution_tf(params, args)#反卷积

    ## Network architectures
    G = select_G(params, args)#'FP'or'Wiener'
    G_optimizer = tf.keras.optimizers.Adam(args.G_lr, beta_1=args.G_beta1)#反卷积的优化器

    ## Construct vgg for perceptual loss VGG感知损失
    if not args.P_loss_weight == 0:
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg_layers = [vgg.get_layer(name).output for name in args.vgg_layers.split(',')]
        vgg_model = tf.keras.Model(inputs=vgg.input, outputs=vgg_layers)
        vgg_model.trainable = False
    else:
        vgg_model = None

    ## Saving the model 
    checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer, G=G, snr=snr)

    max_to_keep = args.max_to_keep#要保存的检查点的数量
    if args.max_to_keep == 0:
        max_to_keep = None
    manager = tf.train.CheckpointManager(checkpoint, directory=args.save_dir, max_to_keep=max_to_keep)

    ## Loading pre-trained model if exists
    if not args.ckpt_dir == None:
        status = checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir, latest_filename=None))
        status.expect_partial() # Silence warnings
        #status.assert_existing_objects_matched() # Only partial load for networks (we don't load the optimizers)
        #status.assert_consumed()

    ## Create summary writer for TensorBoard
    summary_writer = tf.summary.create_file_writer(args.save_dir)

    ## Dataset
    train_ds = iter(train_dataset_sim(params['out_width'], params['load_width'], args))
    test_ds  = list(test_dataset_sim(params['out_width'], params['load_width'], args).take(1))

    ## Do training
    for step in range(args.steps):
        start = time.time()
        if step % args.save_freq == 0:
            print('Saving', flush=True)
            manager.save()
        if step % args.log_freq == 0:
            print('Logging', flush=True)
            test_batch = test_ds[0]
            img = test_batch[0]
            gt_img = test_batch[1]
            log(img, gt_img, G, snr, vgg_model, summary_writer, step, params, args)
        for _ in range(args.Phase_iters):
            img_batch = next(train_ds)
            img = img_batch[0]
            gt_img = img_batch[1]
            train_step('Phase', img, gt_img, G, G_optimizer, snr, vgg_model, params, args)
        for _ in range(args.G_iters):
            img_batch = next(train_ds)
            img = img_batch[0]
            gt_img = img_batch[1]
            train_step('G', img, gt_img,  G, G_optimizer, snr, vgg_model, params, args)
        print("Step time: {}\n".format(time.time() - start), flush=True)


## Entry point
def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
