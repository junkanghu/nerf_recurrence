from .networks import pts_encoding, MLP
import os
import torch
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):  # i=-1时代表不用positional encoding
    if i == -1:  # nn.Identity()继承的是nn.Module，会将输入参数原封不动返回
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)  # 传参时list用*，dict用**
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def batchify_net(net, embeded_all, chunk):
    out = []
    for i in range(0, int(embeded_all.shape[0]), chunk):
        out.append(net(embeded_all[i:i + chunk]))
    return torch.cat(out, dim=0)


def run_network(pts, view_dir, fn, embed, embed_view, net_chunk):
    sh = list(pts.shape[:2])
    view_dir = view_dir[:, None].expand(pts.shape)
    pts = pts.reshape([-1, pts.shape[-1]])
    embeded = embed(pts)
    view_dir = view_dir.reshape([-1, view_dir.shape[-1]])
    embeded_view = embed_view(view_dir)
    embeded_all = torch.cat([embeded, embeded_view], dim=-1)
    out = batchify_net(fn, embeded_all, net_chunk)
    return out.reshape(sh + [out.shape[-1]])


def creat_nerf(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = (opt.fre_loc * 2 + 1) * 3
    input_view_dim = (opt.fre_view * 2 + 1) * 3
    embed = pts_encoding(opt.fre_loc).embed()
    embed_view = pts_encoding(opt.fre_view).embed()
    model = MLP(input_dim, input_view_dim, layer=opt.net_layer, nc=opt.net_nc).to(device)
    model_fn = MLP(input_dim, input_view_dim, layer=opt.net_layer, nc=opt.net_nc).to(device)
    network_query_fn = lambda pts, view_dirs, fn: run_network(pts, view_dirs, fn, embed=embed,
                                                              embed_view=embed_view, net_chunk=opt.net_chunk)
    # define optimizer
    param = list(model.parameters()) + list(model_fn.parameters())
    optimizer = torch.optim.Adam(params=param, lr=opt.lr, betas=(0.9, 0.999))
    # load checkpoints
    global_step = 0
    result_dir = os.path.join(opt.out_dir, opt.data_type, opt.object)
    os.makedirs(result_dir, exist_ok=True)
    content = os.listdir(result_dir)
    content = sorted([i for i in content if "tar" in i])
    print('found checkpoints', content)
    if len(content) > 0 & opt.reload:
        model_name = content[-1]
        ckpt_dir = os.path.join(result_dir, model_name)
        ckpt = torch.load(ckpt_dir)
        global_step = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_dict'])
        model.load_state_dict(ckpt["model_dict"])
        model_fn.load_state_dict(ckpt["model_fn_dict"])
        print("checkpoint reloaded from {}".format(ckpt_dir))

    # creat text file for configurations
    config_dir = os.path.join(result_dir, 'config.txt')
    with open(config_dir, 'w') as file:
        for arg in sorted(vars(opt)):
            attr = getattr(opt, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # define dict
    render_kwargs_train = {"network_query": network_query_fn,
                           "model": model, "model_fn": model_fn,
                           "perturb": opt.perturb, "noise_std": opt.noise_std,
                           "ndc": True, "N_samples": opt.N_samples,
                           'N_importance': opt.N_importance}
    if opt.data_type != 'llff' or opt.no_ndc:
        print('no ndc')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = opt.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False  # test时均匀采样
    render_kwargs_train['noise_std'] = 0.  # test时不对sigma加噪声
    return global_step, optimizer, render_kwargs_test, render_kwargs_test, result_dir


# def initialize(opt, optimizer, model, model_fn):
#     global_step = 0
#     result_dir = os.path.join(opt.out_dir, opt.data_type, opt.object)
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir, exist_ok=True)
#         return global_step
#     content = os.listdir(result_dir)
#     content = sorted([i for i in content if "tar" in i])
#     print('found checkpoints', content)
#     if len(content) > 0 & opt.reload:
#         model_name = content[-1]
#         ckpt_dir = os.path.join(result_dir, model_name)
#         ckpt = torch.load(ckpt_dir)
#         global_step = ckpt['global_step']
#         optimizer.load_state_dict([ckpt["optimizer"]])
#         model.load_state_dict([ckpt["model"]])
#         model_fn.load_state_dict(ckpt["model_fn"])
#         print("checkpoint reloaded from {}".format(ckpt_dir))
#
#     # creat text file for configurations
#     config_dir = os.path.join(result_dir, 'config.txt')
#     with open(config_dir, 'w') as file:
#         for arg in sorted(vars(opt)):
#             attr = getattr(opt, arg)
#             file.write('{} = {}\n'.format(arg, attr))
#     return global_step, optimizer, model, model_fn
