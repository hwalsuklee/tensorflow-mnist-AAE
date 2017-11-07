import argparse
import prior_factory as prior
import matplotlib.pyplot as plt
import numpy as np

"""parsing and configuration"""
def parse_args():
    desc = "Test prior factory"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--prior_type', type=str, default='mixGaussian',
                        choices=['mixGaussian', 'swiss_roll', 'normal'],
                        help='The type of prior', required=True)

    return parser.parse_args()

"""create an N-bin discrete colormap from the specified input map"""
def discrete_cmap(N, base_cmap=None):
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

"""main"""
def main():
    # parse arguments
    args = parse_args()

    # parameters
    batch_size = 10000
    n_dim = 2

    # get samples from prior
    if args.prior_type=='mixGaussian':
        z_id_ = np.random.randint(0,10,size=[batch_size])
        z = prior.gaussian_mixture(batch_size,n_dim, label_indices=z_id_)
    elif args.prior_type=='swiss_roll':
        z_id_ = np.random.randint(0,10,size=[batch_size])
        z = prior.swiss_roll(batch_size,n_dim, label_indices=z_id_)
    elif args.prior_type=='normal':
        z, z_id_ = prior.gaussian(batch_size,n_dim, use_label_info=True)
    else:
        raise Exception("[!] There is no option for " + args.prior_type)

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=z_id_, marker='o', edgecolor='none', cmap=discrete_cmap(10, 'jet'))
    plt.colorbar(ticks=range(10))
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([-4.5, 4.5])
    axes.set_ylim([-4.5, 4.5])
    plt.show()

if __name__ == '__main__':
    main()