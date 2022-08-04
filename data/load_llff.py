import imageio
import torch
import os
import numpy as np
import cv2


def ndc_t(rays_o, rays_d, hwf, near):
    h, w, focal = hwf
    t = -(near + rays_o[:, 2]) / rays_d[:, 2]
    rays_o = rays_o + t[:, None] * rays_d
    ndc_o = torch.stack([-focal / (w / 2.) * rays_o[:, 0] / rays_o[:, 2],
                         -focal / (h / 2.) * rays_o[:, 1] / rays_o[:, 2],
                         1. + 2. * near / rays_o[:, 2]], dim=-1)
    ndc_d = torch.stack([-focal / (w / 2.)*(rays_d[:, 0] / rays_d[:, 2] - rays_o[:, 0] / rays_o[:, 2]),
                         -focal / (h / 2.)*(rays_d[:, 1] / rays_d[:, 2] - rays_o[:, 1] / rays_o[:, 2]),
                         -2. * near / rays_o[:, 2]], dim=-1)
    return ndc_o, ndc_d


def _minify(basedir, factors=None, resolutions=None):
    if resolutions is None:
        resolutions = []
    if factors is None:
        factors = []
    from subprocess import check_output
    need_load = False
    for f in factors:
        img_dir = os.path.join(basedir, 'images_%d' % f)
        if not os.path.exists(img_dir):
            need_load = True
    for f in resolutions:
        img_dir = os.path.join(basedir, 'images_{}x{}'.format(f[0], f[1]))
        if not os.path.exists(img_dir):
            need_load = True
    if not need_load:
        return
    print('minifying')
    for f in factors + resolutions:
        if isinstance(f, int):
            resize_arg = str(100. / f)
            img_dir = os.path.join(basedir, 'images_%d' % f)
        else:
            resize_arg = '{}x{}'.format(f[0], f[1])
            img_dir = os.path.join(basedir, 'images_{}'.format(resize_arg))
        if os.path.exists(img_dir):
            continue
        os.mkdir(img_dir)
        current = os.getcwd()
        ori_dir = os.path.join(basedir, 'images')
        all_file = sorted(os.listdir(ori_dir))
        img_file = [f for f in all_file if any([f.endswith(i) for i in ['png', 'PNG', 'jpeg', 'jpg', 'JPG']])]
        ext = img_file[0].split('.')[-1]
        check_output('cp {}/* {}'.format(ori_dir, img_dir), shell=True)
        args = ' '.join(['mogrify', '-resize', resize_arg, 'format', 'png', '*.{}'.format(ext)])
        os.chdir(img_dir)
        check_output(args, shell=True)
        os.chdir(current)
        if ext != 'png':
            check_output('rm {}/*.{}'.format(img_dir, ext), shell=True)
            print('duplicate removed')
        print('done')


def get_rays(h, w, k, c2w):
    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - k[0][2]) / k[0][0],
                     -(j - k[1][2]) / k[1][1],
                     -np.ones_like(i)
                     ], axis=-1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:, 3], rays_d.shape)
    return np.stack([rays_o, rays_d], axis=0)

def get_rays_torch(h, w, f, c2w):
    i, j = torch.meshgrid(torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32), indexing='xy')
    dirs = torch.stack(
                    [(i - w * .5) / f,
                    -(j - h * .5) / f,
                    -torch.ones_like(i)]
                     , dim=-1)
    rays_d = torch.sum(c2w[:, :3] * dirs[..., None, :], dim=-1)
    rays_o = torch.broadcast_to(c2w[:, 3], rays_d.shape)
    return torch.stack([rays_o, rays_d], dim=0)


def normalize(vec):
    return vec / np.linalg.norm(vec)


def view_matrix(z, up, pos):
    vec2 = z
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, pos], axis=-1)


def pose_avg(poses):
    hwf = poses[0, :3, 4:]
    center = poses[:, :3, 3].mean(0)
    z = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    return np.concatenate([view_matrix(z, up, center), hwf], axis=-1)


def pose_recenter(poses):
    poses_ = poses
    bottom = np.array([0., 0., 0., 1.], dtype=np.float32).reshape([1, 4])
    c2w = pose_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], axis=0)
    bottom_all = np.tile(bottom[None, :], [poses_.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom_all], axis=1)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_


def get_spiral_poses(c2w, up, tt, focal, n_rots, n_pose, z_rate=.5):
    poses = []
    hwf = c2w[:, -1:]
    c2w = c2w[:, :4]
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_pose + 1)[:-1]:
        sita = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.], dtype=np.float32)
        pos = c2w.dot(tt * sita)
        z = normalize(pos - c2w.dot(np.array([0., 0., -focal, 1.], dtype=np.float32)))
        poses.append(np.concatenate([view_matrix(z, up, pos), hwf], -1))
    return np.stack(poses, 0).astype(np.float32)


class load_llff:
    def __init__(self, opt):
        self.opt = opt
        # define directory
        data_dir = opt.data_dir  # D:\A_run\Datasets\nerf_data\nerf_example_data\nerf_llff_data
        object_dir = os.path.join(data_dir, opt.object)
        down_dir = os.path.join(object_dir, ('images_%d' % opt.down_factor) if opt.down_factor != 1 else 'images')
        # minify
        _minify(object_dir, factors=[opt.down_factor])
        # get_poses
        pose = np.load(os.path.join(object_dir, 'poses_bounds.npy'))
        poses = pose[:, :15].reshape([-1, 3, 5])
        poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1).astype(np.float32)
        bds = pose[:, 15:].astype(np.float32)
        print('poses loaded', 'poses shape:', poses.shape, 'bounding depth shape:', bds.shape)
        print("bounding depth", bds.min(), bds.max())
        # preprocess poses
        sc = 1. if opt.bd_factor is None else 1. / (bds.min() * opt.bd_factor)
        bds *= sc
        poses[:, :3, 3] *= sc
        poses = pose_recenter(poses)
        print('poses re-centered')
        # bounding depth
        if not opt.no_ndc:
            near = 0.
            far = 1.
        else:
            near = bds.min() * .9
            far = bds.max() * 1.
        self.near = near
        self.far = far
        print('near:', self.near, "far:", self.far)
        # get images
        all_file = sorted(os.listdir(down_dir))
        img_file = [os.path.join(down_dir, file) for file in all_file if any([file.endswith(mat) \
                                                                              for mat in
                                                                              ['png', 'PNG', 'jpg', 'JPG', 'jpeg']])]
        # img = [cv2.imread(i, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255. for i in img_file]
        img = [imageio.imread(i)[..., :3] / 255. for i in sorted(img_file)]
        imgs = np.stack(img, axis=0)[:, None].astype(np.float32)
        print('images loaded, shape', imgs.shape)
        # get down-sampled poses
        poses[:, :2, 4:] = np.array([img[0].shape[:2]], dtype=np.float32).reshape([2, 1])
        poses[:, 2, 4] = poses[:, 2, 4] * 1. / opt.down_factor
        h, w, focal = poses[0, :3, -1]
        self.hwf = [int(h), int(w), focal]
        self.K = np.array([[focal, 0, 0.5 * w],
                           [0, focal, 0.5 * h],
                           [0, 0, 1]], dtype=np.float32)
        print("H W focal", h, w, focal)
        # get validation poses
        if opt.llff_holdout > 0:
            self.i_val = np.arange(int(poses.shape[0]))[::opt.llff_holdout]
            print('Auto LLFF holdout', opt.llff_holdout)
        else:
            c2w = pose_avg(poses)
            dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), axis=-1)  # 取translation最小的pose
            self.i_val = np.argmin(dists)
            print('holdout view:', self.i_val)
        self.i_train = np.array([i for i in np.arange(int(poses.shape[0])) if i not in self.i_val])
        # get test poses
        if not opt.spherify:
            c2w = pose_avg(poses)  # 几乎与单位矩阵相同，因为这里的poses已经将poses_avg作为世界坐标系原点
            print("recentered c2w\n", c2w[:, :4])
            up = normalize(poses[:, :3, 1].sum(0))
            close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
            dt = 0.75
            focal = 1. / ((1. - dt) / close_depth + dt / inf_depth)
            n_rots = 2
            n_pose = 120
            tt = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
            tt = np.array(list(tt) + [1.], dtype=np.float32)
            self.test_poses = get_spiral_poses(c2w, up, tt, focal, n_rots, n_pose)

        # get rays
        rays = np.stack([get_rays(h, w, self.K, c2w) for c2w in poses[self.i_train]], axis=0)  # [17, 2, h, w, 3]
        print('rays generated')
        if imgs is not None:
            rays = np.concatenate([rays, imgs[self.i_train]], axis=1)
        # divide
        rays = np.transpose(rays, [0, 2, 3, 1, 4]).astype(np.float32)  # N, H, W, 3, 3
        self.rays_train = rays.reshape([-1, rays.shape[-2], 3])
        self.val_poses = poses[self.i_val].astype(np.float32)
        self.test_poses = self.test_poses.astype(np.float32)

    def forward(self):
        return {'rays_train': self.rays_train, 'val_poses': self.val_poses,
                'test_poses': self.test_poses,  'i_train': self.i_train,
                'i_val': self.i_val, 'near': self.near, 'far': self.far, 'hwf': self.hwf}
