import os, losses, utils, glob
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import models.transformation as transformation
from models.TransMorph_DWin_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_DWin_bspl as TransMorph_bspl
import torch.nn.functional as F
import digital_diffeomorphism as dd

def main():
    atlas_dir = 'C:/Junyu_Files/IXI/atlas.pkl'
    test_dir = 'C:/Junyu_Files/IXI/Test/'
    weights = [1, 1]
    Dwin = [7, 5, 3]
    model_idx=-1
    model_folder = 'TransMorphBSplineDWinFull_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/' + model_folder[:-1] + '.csv'):
        os.remove('Quantitative_Results/' + model_folder[:-1] + '.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'NDV'+','+'all_jac', 'Quantitative_Results/' + model_folder[:-1]+'')
    H, W, D = 160, 192, 224
    config = CONFIGS_TM['TransMorphBSpline']
    config.img_size = (H, W, D)
    config.out_size = (H, W, D)
    config.dwin_kernel_size = (Dwin[0], Dwin[1], Dwin[2])
    config.window_size = (H // 32, W // 32, D // 32)
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_seg_oh = F.one_hot(x_seg.long().cuda(), 46).float().squeeze(1).permute(0, 4, 1, 2, 3).cuda()
            output = model((x, y))
            flow = output[2]
            with torch.cuda.device(GPU_iden):
                def_out = transformation.warp(x_seg_oh.cuda().float(), output[2].cuda(), interp_mode='bilinear')
            def_out = torch.argmax(def_out, dim=1, keepdim=True)
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            mask = x_seg.cpu().detach().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
            mask = mask > 0
            disp_field = flow.cpu().detach().numpy()[0]
            trans_ = disp_field + dd.get_identity_grid(disp_field)
            jac_dets = dd.calc_jac_dets(trans_)
            non_diff_voxels, non_diff_tetrahedra, non_diff_volume = dd.calc_measurements(jac_dets, mask)
            total_voxels = np.sum(mask)
            jac_det = non_diff_volume / total_voxels * 100
            jac_det_all = np.sum((jac_dets['all J_i>0'] <= 0)) / np.prod(mask.shape) * 100
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line + ',' + str(jac_det) + ',' + str(jac_det_all)
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1] + '')
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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