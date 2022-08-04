from .load_llff import load_llff


def creat_dataset(opt):
    # if opt.data_type == 'llff':
    dataset = load_llff(opt)
    # elif opt.data_type == 'LINEMOD':
    #     dataset = load_LINEMOD(opt)
    # elif opt.data_type == 'blender':
    #     dataset = load_blender(opt)
    # else:
    #     dataset = load_deepvoxels(opt)
    return dataset.forward()

