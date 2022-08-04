import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.load_llff import ndc_t, get_rays_torch
from tqdm import tqdm
import imageio
DEBUG = True
img2mse = lambda predict, label: torch.mean((predict - label) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
tobyte = lambda x : (np.clip(x, 0, 1) * 255.).astype(np.uint8)


class MLP(nn.Module):
    def __init__(self, input_ch, input_view_ch, layer=8, nc=256):
        super(MLP, self).__init__()
        self.input_ch = input_ch
        self.input_view_ch = input_view_ch
        self.layer = layer
        self.pts_linear = [nn.Linear(input_ch, nc)] + [nn.Linear(nc, nc) if i != layer // 2  \
                           else nn.Linear(input_ch + nc, nc) for i in range(layer - 1)]
        self.pts_linear = nn.Sequential(*self.pts_linear)
        self.feature_linear = nn.Linear(nc, nc)
        self.alpha_linear = nn.Linear(nc, 1)
        self.view_linear = nn.Linear(input_view_ch + nc, nc // 2)
        self.rgb_linear = nn.Linear(nc // 2, 3)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_pts, input_view = torch.split(x, [self.input_ch, self.input_view_ch], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.pts_linear):
            if i == self.layer // 2 + 1:
                h = torch.cat([h, input_pts], dim=-1)
            h = self.pts_linear[i](h)
            h = self.relu(h)
        alpha = self.alpha_linear(h)
        h = self.feature_linear(h)
        feature_view = self.relu(self.view_linear(torch.cat([h, input_view], dim=-1)))
        rgb = self.sigmoid(self.rgb_linear(feature_view))
        return torch.cat([rgb, alpha], dim=-1)
        
        
class pts_encoding:
    def __init__(self, fre):
        self.fre = fre
        
    def forward(self, x):
        cof = 2 ** torch.linspace(0., self.fre-1, self.fre)
        total = [x]
        for fre in cof:
            total.append(torch.sin(fre * x))
            total.append(torch.cos(fre * x))
        return torch.cat(total, dim=-1)

    def embed(self):
        return self.forward


def render(ray, hwf, chunk, near, far, ndc, **kwargs):
    if len(ray.shape) > 3:
        rays_o, rays_d = ray[0], ray[1]
    else:
        rays_o, rays_d = ray[:, 0], ray[:, 1]
    view_dir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dir = view_dir.reshape([-1, 3]).float()
    sh = rays_o.shape
    rays_o = rays_o.reshape([-1, 3]).float()
    rays_d = rays_d.reshape([-1, 3]).float()
    if ndc:
        rays_o, rays_d = ndc_t(rays_o, rays_d, hwf, 1.)
    near, far = near * torch.ones_like(rays_o[:, :1]), far * torch.ones_like(rays_o[:, :1])
    ray_group = torch.cat([rays_o, rays_d, view_dir, near, far], dim=-1)
    all_out = batchify_ray(ray_group, chunk, **kwargs)
    for k in all_out:
        k_sh = list(sh[:-1]) + list(all_out[k].shape[1:])
        all_out[k] = all_out[k].reshape(k_sh)
    return all_out


def render_path(test_poses, hwf, chunk, test_dir, near, far, ndc, **kwargs):
    h, w, f = hwf[0], hwf[1], hwf[2]
    rgbs = []
    disps = []
    for i, c2w in enumerate(tqdm(test_poses)):
        ray = get_rays_torch(h, w, f, c2w)
        out = render(ray, hwf, chunk, near, far, ndc, **kwargs)
        rgb, disp = out['rgb'].cpu().numpy(), out['disp'].cpu().numpy()
        rgbs.append(rgb)
        disps.append(disp)
        if i == 0:
            print('shape of image:', rgb.shape)
        save_dir = os.path.join(test_dir, '{:04d}.png'.format(i + 1))
        imageio.imwrite(save_dir, tobyte(rgb))
    rgbs = np.stack(rgbs, axis=0)
    disps = np.stack(disps, axis=0)
    return rgbs, disps


def render_rays(rays, perturb, N_samples, network_query, model, model_fn,
                N_importance, noise_std, lindisp=False):
    rays_o, rays_d, view_dir, near, far = rays[:, :3], rays[:, 3:6], rays[:, 6:9], rays[:, -2:-1], rays[:, -1:]
    t_vals = torch.linspace(0., 1., N_samples)
    if not lindisp:
        z_vals = near + (far - near) * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    if perturb:
        z_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        low = torch.cat([z_vals[:, :1], z_mid], dim=-1)
        up = torch.cat([z_mid, z_vals[:, -1:]], dim=-1)
        t_rand = torch.rand(low.shape)
        z_vals = low + (up - low) * t_rand
    pts = rays_o[:, None] + z_vals[..., None] * rays_d[:, None]
    # positional encoding
    raw = network_query(pts, view_dir, model)
    rgb, depth, disp, acc, weights = raw2output(raw, z_vals, rays_d, noise_std)
    if N_importance > 0:
        rgb0, depth0, disp0, acc0 = rgb, depth, disp, acc
        z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        z_samples = sample_pdf(weights[:, 1:-1], z_vals_mid, det=(perturb == False), N_importance=N_importance)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts = rays_o[:, None] + z_vals[..., None] * rays_d[:, None]
        raw = network_query(pts, view_dir, model_fn)
        rgb, depth, disp, acc, weights = raw2output(raw, z_vals, rays_d, noise_std)
    ret = {'rgb': rgb, 'depth': depth, 'disp': disp, 'acc': acc,
           'rgb0': rgb0, 'depth0': depth0, 'disp0': disp0, 'acc0': acc0}
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print('there exists nan or inf numbers')
    return ret

# no problem
def raw2output(raw, z_vals, rays_d, noise_std):
    raw2alpha = lambda dist, sig, act=F.relu: 1. - torch.exp(-dist * act(sig))
    rgb, sigma = raw[..., :3], raw[..., -1]
    if noise_std > 0.:
        noise = torch.randn(sigma.shape) * noise_std
        sigma += noise
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[:, :1].shape)], dim=-1)
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
    alpha = raw2alpha(dists, sigma)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, dim=-1)
    return rgb_map, depth_map, disp_map, acc_map, weights

# no problem
def sample_pdf(weights, bins, det, N_importance):
    weights = weights + 1e-5
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    pdf = torch.cat([torch.zeros_like(weights[:, :1]), weights], dim=-1)
    cdf = torch.cumsum(pdf, dim=-1)
    # Inverse Sample
    if not det:
        z_rand = torch.rand([weights.shape[0], N_importance])
    else:
        t = torch.linspace(0., 1., N_importance)
        z_rand = t.expand([weights.shape[0], N_importance])
    z_rand = z_rand.contiguous()
    u = torch.searchsorted(cdf, z_rand, right=True)
    u = u.contiguous()
    lower = torch.max(u - 1, torch.zeros_like(u))
    upper = torch.min(u, int(cdf.shape[-1] - 1) * torch.ones_like(u))
    inds = torch.stack([lower, upper], dim=-1)
    matched_shape = [u.shape[0], N_importance, cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (z_rand - cdf_g[..., 0]) / denom
    z_vals = bins_g[..., 0] + (bins_g[..., 1] - bins_g[..., 0]) * t
    return z_vals


def batchify_ray(ray_group, chunk, **kwargs):
    all_out = {}
    for step in range(0, int(ray_group.shape[0]), chunk):
        all_ret = render_rays(ray_group[step: step + chunk], **kwargs)
        for k in all_ret:
            if k not in all_out:
                all_out[k] = []
            all_out[k].append(all_ret[k])
    all_out = {k: torch.cat(all_out[k], dim=0) for k in all_out}
    return all_out
