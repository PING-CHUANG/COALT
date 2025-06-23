import os

from utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *

import importlib

from utils.focal_loss import FocalLoss

import loralib as lora
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    #settings.use_lmdb = True
    for name in name_list:
        assert name in ["GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "hyper16"]
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(COCO_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(COCO(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "hyper16":
            if settings.use_lmdb:
                print("Building hyper16 from lmdb")
                datasets.append(Hyper16_lmdb(settings.env.hyper16_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(Hyper16(settings.env.hyper16_dir, image_loader=image_loader))
    return datasets

def slt_collate(batch):
    ret = {}
    for k in batch[0].keys():
        here_list = []
        for ex in batch:
            here_list.append(ex[k])
        ret[k] = here_list
    return ret

class SLTLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        if collate_fn is None:
            collate_fn = slt_collate

        super(SLTLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def run(settings):

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importimport_module("config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Build dataloaders
    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir
    bins = cfg.MODEL.BINS
    search_size = cfg.DATA.SEARCH.SIZE
    # Create network
    elif settings.script_name == "COALT":
        net = COALT(cfg)
        dataset_train = sequence_sampler.SequenceSampler(
            datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, hyper_read_image),
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_GAP, max_interval=cfg.DATA.MAX_INTERVAL,
            num_search_frames=cfg.DATA.SEARCH.NUMBER, num_template_frames=1,
            frame_sample_mode='random_interval',
            prob=cfg.DATA.INTERVAL_PROB)
        loader_train = SLTLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.TRAIN.NUM_WORKER,
                                 shuffle=False, drop_last=True)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

# lora update
    if cfg.TRAIN.LORA_RANK > 0:
        lora.mark_only_lora_as_trainable(net, bias='lora_only')
        print("Use LoRA in Transformer encoder, loar_rank: ", settings.lora_rank)
    else:
        print("Do not use LoRA in Transformer encoder, train all parameters.")
    learnable_parameters = count_trainable_parameters(net)
    print("learnable_parameters", learnable_parameters)  # 19,157,504
    print("ratio of learnable_parameters", learnable_parameters / 19157504)

    # Loss functions and Actors
    if settings.script_name == "COALT":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'ce': ce_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': cfg.TRAIN.focal_WEIGHT, 'ce': cfg.TRAIN.ce_WEIGHT.}
        actor = COALTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg, bins=bins, search_size=search_size)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # optimizer = create_optimizer(net, cfg)
    # lr_scheduler, _ = create_scheduler(cfg, optimizer)

    use_amp = getattr(cfg.TRAIN, "AMP", False)
    if settings.script_name == "COALT":
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
        

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True, load_previous_ckpt=True)
    # trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)

