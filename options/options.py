import configargparse


def get_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--N_iters', type=int, help='total iteraions for training')
    parser.add_argument('--data_type', type=str, help='llff, blender, LINEMOD, deepvoxels')
    parser.add_argument('--object', type=str, help='different objects in a dataset, e.g. fern')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--out_dir', type=str, help='directory for results')
    parser.add_argument('--reload', default=True, help='whether reload the trained parameters')
    parser.add_argument('--N_importance', type=int, default=0, help='extra number of sampled points')
    parser.add_argument('--N_batch_rays', type=int, default=0, help='number of rays trained in a iteration')
    parser.add_argument('--N_samples', type=int, default=0, help='number of extra sampled ponits')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=250, help='decay rate for learning rate')
    parser.add_argument('--fre_loc', type=int, default=10, help='positional encoding for locations')
    parser.add_argument('--fre_view', type=int, default=4, help='positional encoding for view directions')
    parser.add_argument('--net_layer', type=int, default=8, help='layers for network in MLP')
    parser.add_argument('--net_nc', type=int, default=256, help='layer channels')
    parser.add_argument('--i_print', type=int, default=100, help='step for printing losses')
    parser.add_argument('--i_video', type=int, default=50000, help='step for forming a video')
    parser.add_argument('--i_weights', type=int, default=10000, help='step for saving models')
    parser.add_argument('--i_test', type=int, default=50000, help='step for testing')
    parser.add_argument('--llff_holdout', type=int, default=0, help='for test data generation')
    parser.add_argument('--down_factor', type=int, default=0, help='down-sample factor for high-resolution images')
    parser.add_argument('--noise_std', type=float, default=0., help='standard deviation for noise')
    parser.add_argument('--no_ndc', action='store_true', help='whether use NDC for training')
    parser.add_argument('--lindisp', type=bool, default=False,
                        help='sample linearly in inverse depth rather than in depth.')
    parser.add_argument('--net_chunk', type=int, default=1024 * 64, help='chunks passed through MLP')
    parser.add_argument('--render_chunk', type=int, default=1024 * 32, help='chunks passed through render')
    parser.add_argument('--bd_factor', type=float, default=.75, help='rescale bounding depth')
    parser.add_argument('--spherify', action='store_true', help='whether using spherical poses for test')
    parser.add_argument('--perturb', type=bool, default=True, help='whether using stratified sample')
    parser.add_argument('--render_only', action='store_true', help='whether test or train')

    return parser.parse_args()
