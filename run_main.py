import argparse
import glob
import os

import numpy as np
import tensorflow as tf

import mnist_data
import prior_factory as prior
import aae
import plot_utils

IMAGE_SIZE_MNIST = 28

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--prior_type', type=str, default='mixGaussian',
                        choices=['mixGaussian', 'swiss_roll', 'normal'],
                        help='The type of prior')

    parser.add_argument('--n_hidden', type=int, default=1000, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=True,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=15,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=15,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=3.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=10000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args

"""main function"""
def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture

    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
    dim_z = 2                      # to visualize learned manifold

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                              # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """

    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    x_id = tf.placeholder(tf.float32, shape=[None, 10], name='input_img_label')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # samples drawn from prior distribution
    z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')
    z_id = tf.placeholder(tf.float32, shape=[None, 10], name='prior_sample_label')

    # network architecture
    y, z, neg_marginal_likelihood, D_loss, G_loss = aae.adversarial_autoencoder(x_hat, x, x_id, z_sample, z_id, dim_img,
                                                                                dim_z, n_hidden, keep_prob)

    # optimization
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "gaussian_MLP_encoder" in var.name]
    ae_vars = [var for var in t_vars if "gaussian_MLP_encoder" or "bernoulli_MLP_decoder" in var.name]

    train_op_ae = tf.train.AdamOptimizer(learn_rate).minimize(neg_marginal_likelihood, var_list=ae_vars)
    train_op_d = tf.train.AdamOptimizer(learn_rate/5).minimize(D_loss, var_list=d_vars)
    train_op_g = tf.train.AdamOptimizer(learn_rate).minimize(G_loss, var_list=g_vars)

    """ training """

    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')

    # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        decoded = aae.decoder(z_in, dim_img, n_hidden)

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
            train_label_ = train_total_data[:, -mnist_data.NUM_LABELS:]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_ids_input = train_label_[offset:(offset + batch_size), :]
                batch_xs_target = batch_xs_input

                # draw samples from prior distribution
                if args.prior_type == 'mixGaussian':
                    z_id_ = np.random.randint(0, 10, size=[batch_size])
                    samples = prior.gaussian_mixture(batch_size, dim_z, label_indices=z_id_)
                elif args.prior_type == 'swiss_roll':
                    z_id_ = np.random.randint(0, 10, size=[batch_size])
                    samples = prior.swiss_roll(batch_size, dim_z, label_indices=z_id_)
                elif args.prior_type == 'normal':
                    samples, z_id_ = prior.gaussian(batch_size, dim_z, use_label_info=True)
                else:
                    raise Exception("[!] There is no option for " + args.prior_type)

                z_id_one_hot_vector = np.zeros((batch_size, 10))
                z_id_one_hot_vector[np.arange(batch_size), z_id_] = 1

                # reconstruction loss
                _, loss_likelihood = sess.run(
                    (train_op_ae, neg_marginal_likelihood),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                               z_id: z_id_one_hot_vector, keep_prob: 0.9})

                # discriminator loss
                _, d_loss = sess.run(
                    (train_op_d, D_loss),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                               z_id: z_id_one_hot_vector, keep_prob: 0.9})

                # generator loss
                for _ in range(2):
                    _, g_loss = sess.run(
                        (train_op_g, G_loss),
                        feed_dict={x_hat: batch_xs_input, x: batch_xs_target, x_id: batch_ids_input, z_sample: samples,
                                   z_id: z_id_one_hot_vector, keep_prob: 0.9})

            tot_loss = loss_likelihood + d_loss + g_loss

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (epoch, tot_loss, loss_likelihood, d_loss, g_loss))

            # if minimum loss is updated or final epoch, plot results
            if epoch%2==0 or min_tot_loss > tot_loss or epoch+1 == n_epochs:
                min_tot_loss = tot_loss
                # Plot for reproduce performance
                if PRR:
                    y_PRR = sess.run(y, feed_dict={x_hat: x_PRR, keep_prob : 1})
                    y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                    PRR.save_images(y_PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")

                # Plot for manifold learning result
                if PMLR and dim_z == 2:
                    y_PMLR = sess.run(decoded, feed_dict={z_in: PMLR.z, keep_prob : 1})
                    y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                    PMLR.save_images(y_PMLR_img, name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

                    # plot distribution of labeled images
                    z_PMLR = sess.run(z, feed_dict={x_hat: x_PMLR, keep_prob : 1})
                    PMLR.save_scattered_image(z_PMLR,id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)