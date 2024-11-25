import torch
import torch.nn.functional as F
from utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor
import os
from ptflops import get_model_complexity_info
import wandb


# Update D1 from >3px to >=3px & >5%
# matlab code:
# E = abs(D_gt - D_est);
# n_err = length(find(D_gt > 0 & E > tau(1) & E. / abs(D_gt) > tau(2)));
# n_total = length(find(D_gt > 0));
# d_err = n_err / n_total;

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    return {"left":x1, "right":x2}

def get_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    #print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    model_size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')

    return model_size

def log_sizes(model):
    ckpt_size = get_model_size(model)
    feat_ckpt_size = get_model_size(model.feature_extraction)
    print("Model Size: %.2f MB" %ckpt_size)
    print("Feature Extraction Size: %.2f MB" %feat_ckpt_size)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 256, 512), input_constructor= prepare_input, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    wandb.log({'Feature_ext_size_MB': feat_ckpt_size,
               'Model_size_MB': ckpt_size,
               "Operations_GMac": float(macs.split()[0]),
               "Parameters_M": float(params.split()[0])
               })




def dict_mean(dict_list, loss_list):
    mean_dict = {}
    losses = loss_list
    for key in dict_list[0].keys():
        if key in losses:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        # else:
        #     mean_dict[key] = sum(d[key][-1] for d in dict_list) / len(dict_list)
    return mean_dict

def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()

# a wrapper to compute metrics for each image individually
def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, masks, *nargs):
        check_shape_for_metric_computation(D_ests, D_gts, masks)
        bn = D_gts.shape[0]  # batch size
        results = []  # a list to store results for each image
        # compute result one by one
        for idx in range(bn):
            # if tensor, then pick idx, else pass the same value
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
                print("masks[idx].float().mean() too small, skip")
            else:
                ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
                results.append(ret)
        if len(results) == 0:
            print("masks[idx].float().mean() too small for all images in this batch, return 0")
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper

@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)
