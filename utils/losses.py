from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from utils.kd_loss_utils import LogL1Loss, LogL1Loss_v2


###############################################################################
""" Loss Function """
###############################################################################


def model_loss(disp_ests, disp_gt, mask, criterion):

    if len(disp_ests) == 2:
        weights = [0.5, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * criterion(disp_est[mask], disp_gt[mask]))
        return sum(all_losses)

    elif len(disp_ests) == 3:
        weights = [0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * criterion(disp_est[mask], disp_gt[mask]))
        return sum(all_losses)

    else:
        weights = [0.5, 0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * criterion(disp_est[mask], disp_gt[mask]))
        return sum(all_losses)



def kd_model_loss(disp_ests, disp_gts, weights, criterion):
    all_losses = []
    only_one_tea_input = len(disp_gts) == 1 and len(disp_ests) > len(disp_gts)
    if only_one_tea_input:
        for disp_est, weight in zip(disp_ests, weights):
            loss_component = weight * criterion(disp_est, disp_gts[0])
            all_losses.append(loss_component)
    else:
        for disp_est, disp_gt, weight in zip(disp_ests, disp_gts, weights):
            all_losses.append(weight * criterion(disp_est, disp_gt))

    return sum(all_losses)



###############################################################################
""" Loss Types """
###############################################################################


class CriterionPairWiseforWholeFeatAfterPool4D(nn.Module):
    def __init__(self, scale=0.5):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool4D, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_disp, total_w, total_h = feat_T.shape[2], feat_T.shape[3], feat_T.shape[4]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        loss=0
        for i in range(total_disp):
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
            loss += self.criterion(maxpool(feat_S[:,:,i,...]), maxpool(feat_T[:,:,i,...]))
        return loss/total_disp


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        #print(total_w, total_h)
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        #print(patch_w, patch_h)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        #print(maxpool(feat_S).shape)
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionFeaSum(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionFeaSum, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        # self.attn1 = Cos_Attn(2048, 'relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss(size_average=True)

    def forward(self, preds, soft):
        cs, ct = preds[1].size(1), soft[1].size(1)
        graph_s = torch.sum(torch.abs(preds[1]), dim=1, keepdim=True) / cs
        graph_t = torch.sum(torch.abs(soft[1]), dim=1, keepdim=True) / ct
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph

class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        #fs_, f_t = equalize_shapes(f_s, f_t)
        loss = self.crit(f_s, f_t)
        return loss

class HintLossSmooth(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLossSmooth, self).__init__()
        self.crit = nn.SmoothL1Loss()

    def forward(self, f_s, f_t):
        #fs_, f_t = equalize_shapes(f_s, f_t)
        loss = self.crit(f_s, f_t)
        return loss

class HintLossHuber(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLossHuber, self).__init__()
        self.crit = nn.HuberLoss()

    def forward(self, f_s, f_t):
        #fs_, f_t = equalize_shapes(f_s, f_t)
        loss = self.crit(f_s, f_t)
        return loss


class KLDLoss(nn.Module):
    """own implementation of KLD loss"""
    def __init__(self):
        super(KLDLoss, self).__init__()
        self.crit = nn.KLDivLoss(reduction="mean")

    def forward(self, soft_pred, soft_gt):
        N, C, H, W = soft_pred.shape

        soft_pred = torch.flatten(soft_pred, 0, -1)
        soft_gt = torch.flatten(soft_gt, 0, -1)

        # # soft_pred[soft_pred < 0.01] = 0
        # # soft_gt[soft_gt < 0.01] = 0
        # # soft_pred = soft_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # # soft_gt = soft_gt.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # zero_tensor = torch.zeros(soft_gt.shape).cuda()
        # #logsoftmax = nn.LogSoftmax(dim=0)
        # #loss = (torch.sum( - soft_gt * (logsoftmax(soft_pred))))/H/W
        # soft_pred = torch.where((soft_pred > 0.01), soft_pred, zero_tensor)
        # soft_gt = torch.where((soft_gt > 0.01), soft_gt, zero_tensor)
        #
        #
        # kld_tensor =  - soft_gt * (torch.log(soft_pred) - torch.log(soft_gt))
        # loss = torch.sum(kld_tensor) / H / W

        kld_tensor =  - soft_gt * (torch.log(soft_pred))
        loss = torch.sum(kld_tensor) / H / W

        #
        # softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        # logsoftmax = nn.LogSoftmax(dim=1)
        # loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H

        # for i in range(0,soft_gt.shape[-1]):
        #     loss += self.crit(soft_pred[:,:,i], soft_gt[:,:,i])

        return loss


class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T.detach()

        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss


class CriterionPixelWise4D(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise4D, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        preds_T.detach()

        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,D,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.permute(0,2,3,4,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S.permute(0,2,3,4,1).contiguous().view(-1,C))))/W/H
        return loss


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            # From Bxdim -> 1xBxdim - Bx1xdim = BxBxdim
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

def equalize_shapes(t1, t2):
    if t1.shape != t2.shape:
        t1 = F.interpolate(t1, (t2.shape[1], t2.shape[2], t2.shape[3]), mode = 'trilinear', align_corners = True)
    return t1, t2

class CriterionPixelWise4D_v2(nn.Module):
	def __init__(self, ignore_index=255, use_weight=True, reduce=True):
         super(CriterionPixelWise4D_v2, self).__init__()
         self.ignore_index = ignore_index
         self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
         if not reduce:
                 print("disabled the reduce.")

	def forward(self, preds_S, preds_T):
         preds_T.detach()

         assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
         N,C,D,W,H = preds_S.shape
         loss = 0
         for i in range(D):
                 softmax_pred_T = F.softmax(preds_T[:, :, i, ...].permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
                 logsoftmax = nn.LogSoftmax(dim=1)
                 loss += (torch.sum(- softmax_pred_T * logsoftmax(preds_S[:, :, i, ...].permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H
         return loss/D

class FocalLoss(nn.Module):
    def __init__(self, focal_coefficient=5.0):
        super(FocalLoss, self).__init__()
        self.focal_coefficient = focal_coefficient

    def forward(self, stud_vol, tea_vol):
        assert stud_vol.shape == tea_vol.shape, 'the output dim of teacher and student differ'

        stud_vol_mean = stud_vol.mean(dim=1)  # N C D H W --> N D H W
        tea_vol_mean = tea_vol.mean(dim=1)
        stud_soft = F.log_softmax(stud_vol_mean, dim=1)
        tea_soft = F.log_softmax(tea_vol_mean, dim=1).exp()
        weight = (1.0 - tea_soft).pow(-self.focal_coefficient).type_as(tea_soft)  # (1-p)**gamma
        loss = -((tea_soft * stud_soft) * weight).sum(dim=1, keepdim=True).mean()  # (1-p)**gamma * cross-entropy
        return loss


class StereoKLD(nn.Module):
    def __init__(self):
        super(StereoKLD, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, stud_vol, tea_vol):
        assert stud_vol.shape == tea_vol.shape, 'the output dim of teacher and student differ'

        stud_vol_mean = stud_vol.mean(dim=1)  # N C D H W --> N D H W
        tea_vol_mean = tea_vol.mean(dim=1)
        studsoft = F.log_softmax(stud_vol_mean, dim=1)
        teasoft = F.log_softmax(tea_vol_mean, dim=1)
        loss = self.kl_loss(studsoft, teasoft)
        return loss


class StereoSimilarity(nn.Module):
    def __init__(self, dim=1):
        super(StereoSimilarity, self).__init__()
        self.cos_loss = nn.CosineSimilarity(dim=dim)

    def forward(self, stud_vol, tea_vol):
        assert stud_vol.shape == tea_vol.shape, 'the output dim of teacher and student differ'
        #print(self.cos_loss(tea_vol, stud_vol))
        loss = 1 - self.cos_loss(tea_vol, stud_vol)
        return loss.mean()


class KR_HLC(nn.Module):
    def __init__(self, kernel_sizes=[16,8,4]):
        super(KR_HLC, self).__init__()
        self.kernel_sizes = kernel_sizes

    def forward(self, fs, ft):
        n,c,d,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in self.kernel_sizes:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool3d(fs, (l,l,l))
            tmpft = F.adaptive_avg_pool3d(ft, (l,l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot

        return loss


class LogL1Loss(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss, self).__init__()
        self.crit = nn.L1Loss()

    def forward(self, f_s, f_t):
        f_s= torch.log(f_s + 0.5)
        f_t= torch.log(f_t + 0.5)
        loss = self.crit(f_s, f_t)
        return loss


class LogL1Loss_v2(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss_v2, self).__init__()
        self.crit = nn.L1Loss(reduction=None)

    def forward(self, f_s, f_t):
        loss = torch.log(torch.abs(f_s - f_t) + 0.5).mean()
        return loss