import glob, sys
import os, losses, utils
import numpy as np
import torch
from natsort import natsorted
from models.TransMorph_ConvMultiDWin_Separate_Chan_SEF_XMLP_TVF_SPT import CONFIGS as CONFIGS_TM
import models.TransMorph_ConvMultiDWin_Separate_Chan_SEF_XMLP_TVF_SPT as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
import torch.nn.functional as F
import timeit
def main():
    test_dir = 'D:/DATA/OASIS/Test/'
    save_dir = 'D:/DATA/OASIS/Submit/submission/task_03/'
    model_idx = -1
    time_steps = 12
    Dwin = [7, 5, 3]
    time_steps = 12
    weights = [1, 1, 1]  # loss weights
    model_folder = 'TransMorphConvMultiDWinSep_SPT_DWin_{}{}{}_half_TVF_{}_ncc_{}_dsc{}_diffusion_{}/'.format(Dwin[0],
                                                                                                          Dwin[1],
                                                                                                          Dwin[2],
                                                                                                          time_steps,
                                                                                                          weights[0],
                                                                                                          weights[1],
                                                                                                          weights[2])
    model_dir = 'experiments/' + model_folder
    H, W, D = 160, 192, 224
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H // 2, W // 2, D // 2)
    config.dwin_kernel_size = (Dwin[0], Dwin[1], Dwin[2])
    config.window_size = (H // 32, W // 32, D // 32)
    model = TransMorph.TransMorphCascadeAd(config, time_steps)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    file_names = glob.glob(test_dir + '*.pkl')
    times = []
    with torch.no_grad():
        stdy_idx = 0
        for data in file_names:
            x, y, x_seg, y_seg = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            x = F.avg_pool3d(x, 2).cuda()
            y = F.avg_pool3d(y, 2).cuda()
            file_name = file_names[stdy_idx].split('\\')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            start = timeit.default_timer()
            output = model((x, y))
            flow = F.interpolate(output.cuda(), scale_factor=2, mode='trilinear') * 2
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            end = timeit.default_timer()
            times.append(end - start)
            print('Time: ', end - start)
            #print('min: {}, max: {}'.format(flow.max(), flow.min()))
            np.savez(save_dir+'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1
        print('Avg. Time: ', np.mean(times[1:]))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()