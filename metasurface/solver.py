import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np

# Initializes parameters used in the simulation and optimization.
def initialize_params(args):
    # Define the `params` dictionary.
    params = dict({})
    args.num_coeffs=8
    params['degrees'] = np.pi / 180
    # Upsampling for Fourier optics propagation
    params['upsample'] = 1
    params['normalize_psf'] = args.normalize_psf  # True or False归一化PSF
    # PSF grid shape.#PSF偏置设置和分成几乘几
    # dim is set to work with the offset PSF training scheme
    if args.offset:
        dim = np.int(2 * 3 - 1 - 1)
    else:
        dim = 5  # <-- TODO: Hack to get image size to be 720 x 720
    psfs_grid_shape = [dim, dim]
    params['psfs_grid_shape'] = psfs_grid_shape

    # Square input image width based on max field angle (20 degrees)
    params['image_width'] = 720

    if args.conv == 'patch_size':
        # Patch sized image for training efficiency
        params['psf_width'] = (params['image_width'] // dim)
        assert (params['psf_width'] % 2 == 0)
        params['hw'] = (params['psf_width']) // 2
        params['load_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + 2 * params['psf_width']
        params['network_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + params['psf_width']
        params['out_width'] = (params['image_width'] // params['psfs_grid_shape'][0])
    elif args.conv == 'full_size':
        # Full size image for inference
        params['psf_width'] = (params['image_width'] // 2)
        print(params['psf_width'])
        assert (params['psf_width'] % 2 == 0)
        params['hw'] = (params['psf_width']) // 2
        params['load_width'] = params['image_width'] + 2 * params['psf_width']
        params['network_width'] = params['image_width'] + params['psf_width']
        params['out_width'] = params['image_width']
    else:
        assert 0

    print('Image width: {}'.format(params['image_width']))
    print('PSF width: {}'.format(params['psf_width']))
    print('Load width: {}'.format(params['load_width']))
    print('Network width: {}'.format(params['network_width']))
    print('Out width: {}'.format(params['out_width']))
    params['a_poisson'] = args.a_poisson  # Poisson noise component泊松
    params['b_sqrt'] = args.b_sqrt  # Gaussian noise standard deviation高斯
    fwhm = np.array([35.0, 34.0, 21.0])  # Screen fwhm
    params['sigma'] = fwhm / 2.355
    calib_fwhm = np.array([15.0, 30.0, 14.0])  # Calibration fwhm
    params['calib_sigma'] = calib_fwhm / 2.355
    # Compute the PSFs on the full field grid without exploiting azimuthal symmetry.在不利用方位对称的情况下计算全场网格上的点扩展函数。
    params['full_field'] = False  # Not currently used

    # Manufacturing considerations.
    params['fab_tolerancing'] = False  # True
    params['fab_error_global'] = 0.03  # +/- 6% duty cycle variation globally (2*sigma)
    params['fab_error_local'] = 0.015  # +/- 3% duty cycle variation locally (2*sigma)

    return params

# Shifts the raw PSF to be centered, cropped to the patch size, and stacked
# along the channels dimension
def shift_and_segment_psf(psf, params):
    # Calculate the shift amounts for each PSF.
    b, h, w = psf.shape
    shifted_psf = psf

    # Reshape the PSFs based on the color channel.
    psf_channels_shape = (params['batchSize'] // (np.size(params['theta_base']) * np.size(params['phi_base'])),
                          np.size(params['theta_base']) * np.size(params['phi_base']),
                        h, w)
    shifted_psf_c_channels = tf.reshape(shifted_psf, shape=psf_channels_shape)
    shifted_psf_c_channels = tf.transpose(shifted_psf_c_channels, perm=(1, 2, 3, 0))

    samples = np.size(params['lambda_base']) // 3
    for j in range(np.size(params['theta_base']) * np.size(params['phi_base'])):
        psfs_j = shifted_psf_c_channels[j, :, :, :]
        for k in range(3):
            psfs_jk = psfs_j[:, :, k * samples: (k + 1) * samples]
            psfs_jk_avg = tf.math.reduce_sum(psfs_jk, axis=2, keepdims=False)
            psfs_jk_avg = psfs_jk_avg[:, :, tf.newaxis]
            if k == 0:
                psfs_channels = psfs_jk_avg
            else:
                psfs_channels = tf.concat([psfs_channels, psfs_jk_avg], axis=2)

        psfs_channels_expanded = psfs_channels[tf.newaxis, :, :, :]
        if j == 0:
            psfs_thetas_channels = psfs_channels_expanded
        else:
            psfs_thetas_channels = tf.concat([psfs_thetas_channels, psfs_channels_expanded], axis=0)

    psfs_thetas_channels = psfs_thetas_channels[:, h // 2 - params['hw']: h // 2 + params['hw'],
                           w // 2 - params['hw']: w // 2 + params['hw'], :]

    # Normalize to unit power per channel since multiple wavelengths are now combined into each channel
    if params['normalize_psf']:
        psfs_thetas_channels_sum = tf.math.reduce_sum(psfs_thetas_channels, axis=(1, 2), keepdims=True)
        psfs_thetas_channels = psfs_thetas_channels / psfs_thetas_channels_sum
    return psfs_thetas_channels


# Rotate PSF (non-SVOLA)
def rotate_psfs(psf, params, rotate=True):
    # psfs_grid_shape = params['psfs_grid_shape']
    # rotations = np.zeros(np.prod(psfs_grid_shape))
    psfs = shift_and_segment_psf(psf, params)
    rot_angle = 0.0
    if rotate:
        angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=np.float32)
        rot_angle = (np.random.choice(angles) * np.pi / 180.0).astype(np.float32)
    rot_angles = tf.fill([np.size(params['theta_base']) * np.size(params['phi_base'])], rot_angle)
    psfs_rot = tfa.image.rotate(psfs, angles=rot_angles, interpolation='NEAREST')
    return psfs_rot


# PSF patches are determined by rotating them into the different patch regions
# for subsequent SVOLA convolution.
def rotate_psf_patches(psf, params):
    psfs_grid_shape = params['psfs_grid_shape']
    rotations = np.zeros(np.prod(psfs_grid_shape))
    psfs = shift_and_segment_psf(psf, params)

    # Iterate through all positions in the PSF grid.
    mid_y = (psfs_grid_shape[0] - 1) // 2
    mid_x = (psfs_grid_shape[1] - 1) // 2
    for i in range(psfs_grid_shape[0]):
        for j in range(psfs_grid_shape[1]):
            r_idx = i - mid_y
            c_idx = j - mid_x

            if params['full_field'] == True:
                index = psfs_grid_shape[0] * j + i
                psf_ij = psfs[index, :, :, :]
            else:
                # Calculate the required rotation angle.
                rotations[i * psfs_grid_shape[0] + j] = np.arctan2(-r_idx, c_idx) + np.pi / 2

                # Set the PSF based on the normalized radial distance.
                psf_ij = psfs[max(abs(r_idx), abs(c_idx)), :, :, :]

            psf_ij = psf_ij[tf.newaxis, :, :, :]

            if (i == 0 and j == 0):
                psf_patches = psf_ij
            else:
                psf_patches = tf.concat([psf_patches, psf_ij], axis=0)

    # Apply the rotations as a batch operation.
    psf_patches = tfa.image.rotate(psf_patches, angles=rotations, interpolation='NEAREST')
    return psf_patches


# Applies Poisson noise and adds Gaussian noise.
def sensor_noise(input_layer, params, clip=(1E-20, 1.)):
    # Apply Poisson noise.
    if (params['a_poisson'] > 0):
        a_poisson_tf = tf.constant(params['a_poisson'], dtype=tf.float32)

        input_layer = tf.clip_by_value(input_layer, clip[0], 100.0)
        p = tfp.distributions.Poisson(rate=input_layer / a_poisson_tf, validate_args=True)
        sampled = tfp.monte_carlo.expectation(f=lambda x: x, samples=p.sample(1), log_prob=p.log_prob,
                                            use_reparameterization=False)
        output = sampled * a_poisson_tf
    else:
        output = input_layer

    # Add Gaussian readout noise.
    gauss_noise = tf.random.normal(shape=tf.shape(output), mean=0.0, stddev=params['b_sqrt'], dtype=tf.float32)
    output = output + gauss_noise

    # Clipping.
    output = tf.clip_by_value(output, clip[0], clip[1])
    return output