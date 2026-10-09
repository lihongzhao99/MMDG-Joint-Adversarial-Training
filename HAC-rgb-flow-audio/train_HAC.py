import torch.utils
import torch.utils.data
from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import os.path as osp
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.test import get_arguments
from dataloader_DG_HAC import HACDOMAIN
import torch.nn.functional as F
import copy

# from mmengine.logging import MMLogger
# logger = MMLogger.get_instance(name="my_logger")
# logger.setLevel('ERROR')

def train_one_step(clip, flow, spectrogram, labels, domain_labels, args,
                   model_video, model_flow, model_audio, mm_cls, criterion, optim):
    labels = labels.cuda()
    domain_labels = domain_labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video.module.backbone.get_feature(clip)
            v_feat = (x_slow.detach(), x_fast.detach())
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_audio:
            a_feat = model_audio.audnet.get_feature(spectrogram)

    embeddings = {}
    unimodal_losses = {}
    if args.use_video:
        if args.use_dsu:
            v_feat = model_video.module.backbone.dsu(v_feat)
        v_feat = model_video.module.backbone.get_predict(v_feat)
        if args.use_dsu:
            v_feat = model_video.module.backbone.dsu(v_feat)
        v_emd = v_norm(model_video.module.backbone.avg_pool(v_feat))
        v_pred = v_cls(v_emd)
        embeddings['v'] = v_emd
        unimodal_losses['v'] = criterion(v_pred, labels)

    if args.use_flow:
        if args.use_dsu:
            f_feat = model_flow.module.backbone.dsu(f_feat)
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())
        if args.use_dsu:
            f_feat = model_flow.module.backbone.dsu(f_feat)
        f_emd = f_norm(model_flow.module.backbone.avg_pool(f_feat))
        f_pred = f_cls(f_emd)
        embeddings['f'] = f_emd
        unimodal_losses['f'] = criterion(f_pred, labels)

    if args.use_audio:
        if args.use_dsu:
            a_feat = model_audio.audnet.dsu(a_feat)
        a_feat = model_audio.audnet.get_predict(a_feat.detach())
        if args.use_dsu:
            a_feat = model_audio.audnet.dsu(a_feat)
        a_emd = a_norm(model_audio.audnet.avg_pool(a_feat))
        a_pred = a_cls(a_emd)
        embeddings['a'] = a_emd
        unimodal_losses['a'] = criterion(a_pred, labels)

    modality_order = [m for m in ('v', 'a', 'f') if m in embeddings]
    fusion_order = modality_order
    if modality_order == ['a', 'f']:
        # Keep the fusion layout used by validation/test for this pair.
        fusion_order = ['f', 'a']
    dropped_embeddings = fixed_modality_dropout(
        [embeddings[m] for m in fusion_order], args.modality_dropout)
    mm_pred = mm_cls(torch.cat(dropped_embeddings, dim=1))
    fusion_loss = criterion(mm_pred, labels)

    modal_adv_loss = labels.new_tensor(0.0, dtype=torch.float)
    local_domain_loss = labels.new_tensor(0.0, dtype=torch.float)
    global_domain_loss = labels.new_tensor(0.0, dtype=torch.float)
    entropies = []

    for modal_idx, modal_name in enumerate(modality_order):
        embedding = embeddings[modal_name]
        modal_logits = modal_disc(
            modal_projectors[modal_name](grad_reverse(embedding, args.alpha_modal)))
        modal_labels = torch.full_like(labels, modal_idx)
        modal_adv_loss = modal_adv_loss + criterion(modal_logits, modal_labels)

        if args.num_domains > 1:
            domain_logits = domain_disc(
                domain_projectors[modal_name](grad_reverse(embedding, args.alpha_domain)))
            local_domain_loss = local_domain_loss + criterion(domain_logits, domain_labels)
            domain_prob = F.softmax(domain_logits, dim=1).clamp_min(1e-12)
            entropies.append(-(domain_prob * domain_prob.log()).sum(dim=1).mean())

    num_modalities = len(modality_order)
    modal_adv_loss = modal_adv_loss / num_modalities

    if args.num_domains > 1:
        local_domain_loss = local_domain_loss / num_modalities
        global_embedding = torch.cat([embeddings[m] for m in fusion_order], dim=1)
        global_domain_logits = domain_disc_global(
            grad_reverse(global_embedding, args.alpha_domain))
        global_domain_loss = criterion(global_domain_logits, domain_labels)
        modality_weights = F.softmax(torch.stack(entropies) / args.weight_temperature, dim=0)
    else:
        modality_weights = torch.full(
            (num_modalities,), 1.0 / num_modalities, device=labels.device)

    # Multiplication by M preserves the original scale of the sum of
    # unimodal classification losses when all modality weights are uniform.
    weighted_unimodal_loss = num_modalities * sum(
        modality_weights[idx] * unimodal_losses[name]
        for idx, name in enumerate(modality_order))

    loss = fusion_loss + args.cls_loss * weighted_unimodal_loss
    loss = loss + args.modal_adv_loss * modal_adv_loss
    if args.num_domains > 1:
        loss = loss + args.domain_adv_loss_local * local_domain_loss
        loss = loss + args.domain_adv_loss_global * global_domain_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    return mm_pred, loss

def validate_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video.module.backbone.get_feature(clip)
            v_feat = (x_slow, x_fast)
            v_feat = model_video.module.backbone.get_predict(v_feat)
            v_emd = model_video.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_emd = model_flow.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio.audnet.get_feature(spectrogram)
            a_feat = model_audio.audnet.get_predict(a_feat)
            a_emd = model_audio.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict = mm_cls(feat)
        loss = criterion(predict, labels)

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video_ema.module.backbone.get_feature(clip)
            v_feat = (x_slow, x_fast)
            v_feat = model_video_ema.module.backbone.get_predict(v_feat)
            v_emd = model_video_ema.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow_ema.module.backbone.get_feature(flow)
            f_feat = model_flow_ema.module.backbone.get_predict(f_feat)
            f_emd = model_flow_ema.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio_ema.audnet.get_feature(spectrogram)
            a_feat = model_audio_ema.audnet.get_predict(a_feat)
            a_emd = model_audio_ema.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict_ema = mm_cls_ema(feat)
        loss_ema = criterion(predict_ema, labels)

    return predict, loss, predict_ema, loss_ema

def test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model_video_ema.module.backbone.get_feature(clip)
            v_feat = (x_slow, x_fast)
            v_feat = model_video_ema.module.backbone.get_predict(v_feat)
            v_emd = model_video_ema.module.backbone.avg_pool(v_feat)
            v_emd = v_norm(v_emd)
        if args.use_flow:
            f_feat = model_flow_ema.module.backbone.get_feature(flow)
            f_feat = model_flow_ema.module.backbone.get_predict(f_feat)
            f_emd = model_flow_ema.module.backbone.avg_pool(f_feat)
            f_emd = f_norm(f_emd)
        if args.use_audio:
            a_feat = model_audio_ema.audnet.get_feature(spectrogram)
            a_feat = model_audio_ema.audnet.get_predict(a_feat)
            a_emd = model_audio_ema.audnet.avg_pool(a_feat)
            a_emd = a_norm(a_emd)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, a_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, a_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, a_emd), dim=1)

        predict_ema = mm_cls_ema(feat)
        loss_ema = criterion(predict_ema, labels)

    return predict_ema, loss_ema

class PredHead(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(PredHead, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self, feat):
        feat = self.layernorm(feat)
        return feat

class ProjectHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    def forward(self, feat):
        return self.projector(feat)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, feat):
        return self.classifier(feat)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, feat):
        return self.classifier(feat)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, alpha):
        ctx.alpha = alpha
        return feat.view_as(feat)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(feat, alpha):
    return GradientReversalLayer.apply(feat, alpha)

def fixed_modality_dropout(embeddings, drop_probability):
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError('modality_dropout must be in [0, 1).')
    if drop_probability == 0.0:
        return embeddings

    batch_size = embeddings[0].size(0)
    device = embeddings[0].device
    keep = torch.bernoulli(torch.full(
        (batch_size, len(embeddings)), 1.0 - drop_probability, device=device))

    # A fusion sample must contain at least one modality.
    empty_rows = keep.sum(dim=1) == 0
    if empty_rows.any():
        row_indices = empty_rows.nonzero(as_tuple=False).flatten()
        modal_indices = torch.randint(len(embeddings), (len(row_indices),), device=device)
        keep[row_indices, modal_indices] = 1.0

    return [embedding * keep[:, idx:idx + 1]
            for idx, embedding in enumerate(embeddings)]

@torch.no_grad()
def update_ema(student, ema, decay):
    student_state = student.state_dict()
    ema_state = ema.state_dict()
    for name, ema_value in ema_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(student_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument('--use_dsu', action='store_true', help="Enable DSU: UNCERTAINTY MODELING FOR OUT-OF-DISTRIBUTION GENERALIZATION")
    parser.add_argument('--ema_beta', type=float, default=0.999)
    parser.add_argument('--modality_dropout', type=float, default=0.5)
    parser.add_argument('--project_out_dim', type=int, default=512)
    parser.add_argument('--alpha_modal', type=float, default=0.1)
    parser.add_argument('--alpha_domain', type=float, default=0.3)
    parser.add_argument('--weight_temperature', type=float, default=1.0)
    parser.add_argument('--cls_loss', type=float, default=1.0)
    parser.add_argument('--modal_adv_loss', type=float, default=0.1)
    parser.add_argument('--domain_adv_loss_local', type=float, default=0.5)
    parser.add_argument('--domain_adv_loss_global', type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    # Modality Settings
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')

    # Dataset & Path Settings
    parser.add_argument('-s','--source_domain', nargs='+', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/HAC/')
    parser.add_argument("--num_class", type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=3)

    # Logging & Results
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestValAcc', type=float, default=0)
    parser.add_argument('--BestTestAcc', type=float, default=0)

    # System Settings
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.num_modals = sum((args.use_video, args.use_audio, args.use_flow))
    args.num_domains = len(args.source_domain)
    if args.num_modals < 2:
        parser.error('At least two modalities must be enabled.')
    if args.weight_temperature <= 0:
        parser.error('--weight_temperature must be positive.')

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load config files and checkpoint
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    batch_size = args.bsz
    input_dim = 0
    embedding_dims = {}
    model_video, model_flow, model_audio = None, None, None
    cfg_video, cfg_flow = None, None

    if args.use_video:
        model_video = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        cfg_video = model_video.cfg
        model_video = torch.nn.DataParallel(model_video)

        v_norm = LayerNorm(2304).cuda()
        v_cls = PredHead(input_dim=2304, out_dim=args.num_class).cuda()

        input_dim += 2304
        embedding_dims['v'] = 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_norm = LayerNorm(2048).cuda()
        f_cls = PredHead(input_dim=2048, out_dim=args.num_class).cuda()

        input_dim += 2048
        embedding_dims['f'] = 2048

    if args.use_audio:
        audio_args = get_arguments()
        model_audio = AVENet(audio_args)
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar", map_location=device)
        model_audio.load_state_dict(checkpoint['model_state_dict'])
        model_audio = model_audio.cuda()

        a_norm = LayerNorm(512).cuda()
        a_cls = PredHead(input_dim=512, out_dim=args.num_class).cuda()

        input_dim += 512
        embedding_dims['a'] = 512

    mm_cls = PredHead(input_dim=input_dim, out_dim=args.num_class).cuda()
    modal_projectors = nn.ModuleDict({
        name: ProjectHead(dim, args.project_out_dim)
        for name, dim in embedding_dims.items()
    }).cuda()
    domain_projectors = nn.ModuleDict({
        name: ProjectHead(dim, args.project_out_dim)
        for name, dim in embedding_dims.items()
    }).cuda()
    modal_disc = Classifier(args.project_out_dim, args.num_modals).cuda()
    domain_disc = DomainClassifier(args.project_out_dim, args.num_domains).cuda()
    domain_disc_global = PredHead(
        input_dim=input_dim,
        out_dim=args.num_domains,
        hidden=args.hidden_dim,
    ).cuda()

    # create save file
    if len(args.source_domain) == 1:
        model_dir_base = "models/JAT/single_source"
        if not os.path.exists(model_dir_base):
            os.makedirs(model_dir_base)
        log_dir_base = "results/JAT/single_source"
        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)
    elif len(args.source_domain) == 2:
        model_dir_base = "models/JAT/multi_source"
        if not os.path.exists(model_dir_base):
            os.makedirs(model_dir_base)
        log_dir_base = "results/JAT/multi_source"
        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)
    else:
        raise ValueError("create file fail!")

    dir_name = []
    if args.use_video:
        dir_name.append('video')
    if args.use_flow:
        dir_name.append('flow')
    if args.use_audio:
        dir_name.append('audio')
    dir_name = '_'.join(dir_name)

    model_dir = osp.join(model_dir_base, dir_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    log_dir = osp.join(log_dir_base, dir_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model_path = osp.join(model_dir, f"{args.source_domain}_to_{args.target_domain}_seed{args.seed}.pt")
    log_path = osp.join(log_dir, f"{args.source_domain}_to_{args.target_domain}_seed{args.seed}.csv")

    criterion = nn.CrossEntropyLoss()

    params = (list(mm_cls.parameters()) + list(modal_projectors.parameters())
              + list(domain_projectors.parameters()) + list(modal_disc.parameters())
              + list(domain_disc.parameters()) + list(domain_disc_global.parameters()))
    if args.use_video:
        params = params + list(model_video.module.backbone.fast_path.layer4.parameters()) + list(model_video.module.backbone.slow_path.layer4.parameters()) + list(v_cls.parameters())
    if args.use_flow:
        params = params + list(model_flow.module.backbone.layer4.parameters()) + list(f_cls.parameters())
    if args.use_audio:
        params = params + list(model_audio.audnet.layer4.parameters()) + list(a_cls.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestValAcc = args.BestValAcc
    BestTestAcc = args.BestTestAcc

    train_dataset = HACDOMAIN(split='train', source=True, domain=args.source_domain, cfg_video=cfg_video, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    validate_dataset = HACDOMAIN(split='test', source=True, domain=args.source_domain, cfg_video=cfg_video, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    if len(args.target_domain) == 1:
        test_dataset = HACDOMAIN(split='test', source=False, domain=args.target_domain, cfg_video=cfg_video, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    else:
        test_dataset1 = HACDOMAIN(split='test', source=False, domain=args.target_domain[0:1], cfg_video=cfg_video, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
        test_dataset2 = HACDOMAIN(split='test', source=False, domain=args.target_domain[1:], cfg_video=cfg_video, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
        test_dataloader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
        test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': validate_dataloader}
    with open(log_path, "a") as f:
        # ema model
        if args.use_video:
            model_video_ema = copy.deepcopy(model_video)
            model_video_ema.eval()
        else:
            model_video_ema = None
        if args.use_flow:
            model_flow_ema = copy.deepcopy(model_flow)
            model_flow_ema.eval()
        else:
            model_flow_ema = None
        if args.use_audio:
            model_audio_ema = copy.deepcopy(model_audio)
            model_audio_ema.eval()
        else:
            model_audio_ema = None
        mm_cls_ema = copy.deepcopy(mm_cls)
        mm_cls_ema.eval()
        # iteration
        for epoch_i in range(1, args.nepochs+1):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                acc_ema = 0
                print(split)
                mm_cls.train(split == 'train')
                modal_projectors.train(split == 'train')
                domain_projectors.train(split == 'train')
                modal_disc.train(split == 'train')
                domain_disc.train(split == 'train')
                domain_disc_global.train(split == 'train')
                if args.use_video:
                    model_video.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                if args.use_audio:
                    model_audio.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (clip, flow, spectrogram, labels, domain_labels) in dataloaders[split]:
                        if split=='train':
                            predict1, loss = train_one_step(
                                clip, flow, spectrogram, labels, domain_labels, args,
                                model_video, model_flow, model_audio, mm_cls,
                                criterion, optim)
                            if args.use_video:
                                update_ema(model_video, model_video_ema, args.ema_beta)
                            if args.use_flow:
                                update_ema(model_flow, model_flow_ema, args.ema_beta)
                            if args.use_audio:
                                update_ema(model_audio, model_audio_ema, args.ema_beta)
                            update_ema(mm_cls, mm_cls_ema, args.ema_beta)
                        else:
                            predict1, loss, predict1_ema, loss_ema = validate_one_step(clip, flow, spectrogram, labels, args, model_video, model_flow, model_audio, mm_cls, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)
                            _, predict_ema = torch.max(predict1_ema.detach().cpu(), dim=1)
                            acc1_ema = (predict_ema == labels).sum().item()
                            acc_ema += int(acc1_ema)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str("Average loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count), acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc_ema / float(count)
                        if currentvalAcc >= BestValAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestValAcc = currentvalAcc
                            save = {
                                'BestEpoch': BestEpoch,
                                'BestValAcc': BestValAcc,
                            }
                            save['mm_cls_ema_state_dict'] = mm_cls_ema.state_dict()
                            if args.use_video:
                                save['model_video_ema_state_dict'] = model_video_ema.state_dict()
                            if args.use_flow:
                                save['model_flow_ema_state_dict'] = model_flow_ema.state_dict()
                            if args.use_audio:
                                save['model_audio_ema_state_dict'] = model_audio_ema.state_dict()
                            torch.save(save, model_path)

                    if split == 'train':
                        f.write(f"epoch {epoch_i}\n")
                        f.write("{:5}, {}, {}\n".format(split, total_loss/float(count), acc/float(count)))
                    else:
                        f.write("{:5}, {}, {}, {}\n".format(split, total_loss / float(count), acc / float(count),
                                                          acc_ema / float(count)))
                    f.flush()

            print('BestEpoch ', BestEpoch)
            print('BestValAcc ', BestValAcc)

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch, BestLoss, BestValAcc))
        f.flush()

        # load best model
        ckt = torch.load(model_path)
        mm_cls_ema.load_state_dict(ckt['mm_cls_ema_state_dict'])
        if args.use_video:
            model_video_ema.load_state_dict(ckt['model_video_ema_state_dict'])
        if args.use_flow:
            model_flow_ema.load_state_dict(ckt['model_flow_ema_state_dict'])
        if args.use_audio:
            model_audio_ema.load_state_dict(ckt['model_audio_ema_state_dict'])

        # test
        print('test')
        if len(args.target_domain) == 1:
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader)) as pbar:
                for (i, (clip, flow, spectrogram, labels, _)) in enumerate(test_dataloader):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                print('TestAcc ', acc/float(count))
                f.write("test , {}, {}\n".format(total_loss/float(count), acc/float(count)))
                f.flush()
        else:
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader1)) as pbar:
                for (i, (clip, flow, spectrogram, labels, _)) in enumerate(test_dataloader1):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                print(f'Test {args.target_domain[0]} ', acc/float(count))
                f.write("test {}, {}, {}\n".format(args.target_domain[0], total_loss/float(count), acc/float(count)))
                f.flush()
            acc = 0
            count = 0
            total_loss = 0
            with tqdm.tqdm(total=len(test_dataloader2)) as pbar:
                for (i, (clip, flow, spectrogram, labels, _)) in enumerate(test_dataloader2):
                    predict1, loss = test_one_step(clip, flow, spectrogram, labels, args, model_video_ema, model_flow_ema, model_audio_ema, mm_cls_ema, criterion)

                    total_loss += loss.item() * batch_size
                    _, predict = torch.max(predict1.detach().cpu(), dim=1)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]
                    pbar.set_postfix_str(
                        "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                loss.item(),
                                                                                                acc / float(count)))
                    pbar.update()

                print(f'Test {args.target_domain[1]} ', acc/float(count))
                f.write("test {}, {}, {}\n".format(args.target_domain[1], total_loss/float(count), acc/float(count)))
                f.flush()

    f.close()
