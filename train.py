import argparse
import torch
# from datasets.HRSC_dataset import HRSCDataset
from datasets.SSDD_dataset import SSDDataset
from datasets.collater import Collater
import torch.utils.data as data
from utils.utils import set_random_seed, count_param
from models.model import RetinaNet
import torch.optim as optim
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import datetime
import torch.nn as nn
from warmup import WarmupLR
import yaml
from pprint import pprint
from eval import evaluate
from Logger import Logger


class Params:
    def __init__(self, project_file):
        self.filename = os.path.basename(project_file)
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

    def info(self):
        return '\n'.join([(f'{key}: {value}') for key, value in self.params.items()])

    def save(self):
        with open(os.path.join(self.params.get('output_path'), f'{self.filename}'), 'w') as f:
            yaml.dump(self.params, f, sort_keys=False)

    def show(self):
        print('=================== Show Params =====================')
        pprint(self.params)


def get_args():
    parser = argparse.ArgumentParser('A Rotation Detector based on RetinaNet by PyTorch.')
    parser.add_argument('--config_file', type=str, default='./configs/retinanet_r50_fpn_{Dataset Name}.yml')
    parser.add_argument('--resume', type=str,
                        # default='{epoch}_{step}.pth',
                        default=None,  # train from scratch
                        help='the last checkpoint file.')
    args = parser.parse_args()
    return args


def train(args, params):
    epochs = params.epoch
    if torch.cuda.is_available():
        if len(params.device) == 1:
            device = params.device[0]
        else:
            print(f'[Info]: Traing with {params.device} GPUs')

    weight = ''
    if args.resume:
        weight = params.output_path + os.sep + params.checkpoint + os.sep + args.resume

    start_epoch = 0
    best_fitness = 0
    fitness = 0
    last_step = 0

    # create folder
    tensorboard_path = os.path.join(params.output_path, params.tensorboard)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    checkpoint_path = os.path.join(params.output_path, params.checkpoint)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    best_checkpoint_path = os.path.join(checkpoint_path, 'best')
    if not os.path.exists(best_checkpoint_path):
        os.makedirs(best_checkpoint_path)

    log_file_path = os.path.join(params.output_path, params.log)
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)

    log = Logger(log_path=os.path.join(params.output_path, params.log), logging_name='R-RetinaNet')
    logger = log.logger_config()
    env_info = params.info()
    logger.info('Config info:\n' + log.dash_line + env_info + '\n' + log.dash_line)

    # save config yaml file
    params.save()

    train_dataset = SSDDataset(root_path=params.data_path, set_name='train', augment=params.augment,
                                classes=params.classes)
    collater = Collater(scales=params.image_size, keep_ratio=params.keep_ratio, multiple=32)
    train_generator = data.DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        num_workers=8,  # 4 * number of the GPU
        collate_fn=collater,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    # Initialize model & set random seed
    set_random_seed(seed=42, deterministic=False)
    model = RetinaNet(params)
    count_param(model)

    # init tensorboardX
    writer = SummaryWriter(tensorboard_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # Optimizer Option
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    # Scheduler Option
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.6, 0.8]], gamma=0.1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

    # Warm-up
    is_warmup = False
    if params.warm_up and args.resume is None:
        print('[Info]: Launching Warmup.')
        scheduler = WarmupLR(scheduler, init_lr=params.warmup_lr, num_warmup=params.warmup_epoch, warmup_strategy='cos')
        is_warmup = True
    if is_warmup is False:
        print('[Info]: Not Launching Warmup.')

    if torch.cuda.is_available() and len(params.device) == 1:
        model = model.cuda(device=device)
    else:
        model = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
        model.cuda()  # put the model on the main card in the condition of the multi-gpus

    if args.resume:
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            last_step = chkpt['step']

            # Load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

            # Load optimizer
            if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(device=device)

            # Load scheduler
            if 'scheduler' in chkpt.keys() and chkpt['scheduler'] is not None:
                scheduler_state = chkpt['scheduler']
                scheduler._step_count = scheduler_state['step_count']
                scheduler.last_epoch = scheduler_state['last_epoch']

            start_epoch = chkpt['epoch'] + 1

            del chkpt

    # start training
    step = max(0, last_step)
    num_iter_per_epoch = len(train_generator)

    head_line = ('%10s' * 8) % ('Epoch', 'Steps', 'gpu_mem', 'cls', 'reg', 'total', 'targets', 'img_size')
    print(('\n' + '%10s' * 8) % ('Epoch', 'Steps', 'gpu_mem', 'cls', 'reg', 'total', 'targets', 'img_size'))
    logger.debug(head_line)

    if is_warmup:
        scheduler.step()
    for epoch in range(start_epoch, epochs):
        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue
        pbar = tqdm(enumerate(train_generator), total=len(train_generator))  # progress bar

        # for each epoch, we set model.eval() to model.train()
        # and freeze backbone BN Layers parameters
        model.train()

        if params.freeze_bn and len(params.device) == 1:
            model.freeze_bn()
        else:
            model.module.freeze_bn()

        for iter, (ni, batch) in enumerate(pbar):

            if iter < step - last_epoch * num_iter_per_epoch:
                pbar.update()
                continue

            optimizer.zero_grad()
            images, annots, image_names = batch['image'], batch['bboxes'], batch['image_name']
            if torch.cuda.is_available():
                if len(params.device) == 1:
                    images, annots = images.cuda(device=device), annots.cuda(device=device)
                else:
                    images, annots = images.cuda(), annots.cuda()
            loss_cls, loss_reg = model(images, annots, image_names)

            # Using .mean() is following Ming71 and Zylo117 repo
            loss_cls = loss_cls.mean()
            loss_reg = loss_reg.mean()

            total_loss = loss_cls + loss_reg

            if not torch.isfinite(total_loss):
                print('[Warning]: loss is nan')
                break

            if bool(total_loss == 0):
                continue

            total_loss.backward()

            # Update parameters

            # if loss is not nan not using grad clip
            # nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            # print batch result
            if len(params.device) == 1:
                mem = torch.cuda.memory_reserved(device=device) / 1E9 if torch.cuda.is_available() else 0
            else:
                mem = sum(torch.cuda.memory_reserved(device=idx) for idx in range(len(params.device))) / 1E9

            s = ('%10s' * 3 + '%10.3g' * 4 + '%10s' * 1) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g' % iter,
                '%.3gG' % mem, loss_cls.item(), loss_reg.item(), total_loss.item(), annots.shape[1],
                '%gx%g' % (int(images.shape[2]), int(images.shape[3])))

            pbar.set_description(s)

            # write loss info into tensorboard
            writer.add_scalars('Loss', {'train': total_loss}, step)
            writer.add_scalars('Regression_loss', {'train': loss_reg}, step)
            writer.add_scalars('Classfication_loss', {'train': loss_cls}, step)

            # write lr info into tensorboard
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr_per_step', current_lr, step)
            step = step + 1

        # Update scheduler / learning rate
        scheduler.step()
        logger.debug(s)

        final_epoch = epoch + 1 == epochs

        # # check the mAP on training set  begin ------------------------------------------------
        # if epoch >= params.evaluate_train_start and epoch % params.val_interval == 0:
        #     test_path = 'train-ground-truth'
        #     train_results = evaluate(
        #         target_size=[params.image_size],
        #         test_path=test_path,
        #         eval_method=args.eval_method,
        #         model=model,
        #         conf=params.score_thr,
        #         device=args.device,
        #         mode='train')
        #
        #     train_fitness = train_results[0]  # Update best mAP
        #     writer.add_scalar('train_mAP', train_fitness, epoch)
        # --------------------------end

        # save model
        # create checkpoint
        chkpt = {'epoch': epoch,
                 'step': step,
                 'best_fitness': best_fitness,
                 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel
                 else model.state_dict(),
                 'optimizer': None if final_epoch else optimizer.state_dict(),
                 'scheduler': {'step_count': scheduler._step_count,
                               'last_epoch': scheduler.last_epoch}
                 }

        # save interval checkpoint
        if epoch % params.save_interval == 0 and epoch >= 30:
            torch.save(chkpt, os.path.join(checkpoint_path, f'{epoch}_{step}.pth'))

        if epoch >= params.evaluation_val_start and epoch % params.val_interval == 0:
            test_path = 'ground-truth'
            model.eval()
            val_mAP, val_Precision, val_Recall = evaluate(model=model,
                                                          target_size=params.image_size,
                                                          test_path=test_path,
                                                          conf=params.score_thr,
                                                          device=device,
                                                          mode='test',
                                                          params=params)

            eval_line = ('%10s' * 7) % ('[%g/%g]' % (epoch, epochs - 1), 'Val mAP:', '%10.3f' % val_mAP,
                                        'Precision:', '%10.3f' % val_Precision,
                                        'Recall:', '%10.3f' % val_Recall)
            logger.debug(eval_line)

            fitness = val_mAP  # Update best mAP

            if fitness > best_fitness:
                best_fitness = fitness

            # write mAP info into tensorboard
            writer.add_scalar('val_mAP', fitness, epoch)

        # save best checkpoint
        if best_fitness == fitness:
            torch.save(chkpt, os.path.join(best_checkpoint_path, 'best.pth'))

        # TensorboardX writer close
    writer.close()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3, 2'  # for multi-GPU
    from utils.utils import show_args
    args = get_args()
    params = Params(args.config_file)
    show_args(args)
    train(args, params)
