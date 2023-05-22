import os
import argparse
import shutil
import builtins
import csv
import random
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import transforms
from composite_adv.attacks import *
from composite_adv.utilities import make_dataloader
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def list_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='wideresnet',
                    help='architecture of model')
parser.add_argument('--model-dir', default='./model-svhn',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--mode', default='natural', type=str, choices=['natural','adv_train_madry','adv_train_trades'],
                    help='specify training mode (natural or adv_train)')
parser.add_argument('--debug', action='store_true',
                    help='Train Only One Epoch and print training images.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--order', default='random', type=str, help='specify the order')
parser.add_argument('--stat-dict', type=str, default=None,
                    help='key of stat dict in checkpoint')
parser.add_argument("--enable", type=list_type, default=(0, 1, 2, 3, 4, 5), help="list of enabled attacks")
parser.add_argument("--power", type=str, default='strong', help="level of attack power")
parser.add_argument("--linf_loss", type=str, default='ce', help="loss for linf-attack, ce or kl")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:9527', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

start_num = 1
iter_num = 5
inner_iter_num = 10

sequence_single = [(0,), (1,), (2,), (3,), (4,), (5,)]
attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-Infinity"]

epoch = 0
best_acc1 = .0
no_improve = 0


def main():
    # settings
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    from composite_adv.utilities import make_model
    model = make_model(args.arch, 'svhn', checkpoint_path=args.checkpoint)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Send to GPU
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                sd = checkpoint['model_state_dict']
                # sd = {k[len('module.'):]: v for k, v in sd.items()}  # Use this if missing key matching
                model.load_state_dict(sd)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                epoch = 0
            if 'best_acc1' in checkpoint:
                best_acc1 = checkpoint['best_acc1']
            if 'no_improve' in checkpoint:
                no_improve = checkpoint['no_improve']

            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
            print('best_accuracy --> ', best_acc1)
            print('No improve --> ', no_improve)
        else:
            raise ValueError("Checkpoint is not a dictionary.")

    # setup data loader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader, train_sampler = make_dataloader('../data', 'svhn', args.batch_size, transform_train,
                                                  train=True, distributed=args.distributed)
    test_loader = make_dataloader('../data', 'svhn', args.batch_size, transform_test,
                                  train=False, distributed=args.distributed)

    train(model, optimizer, criterion, train_loader, train_sampler, test_loader, args, ngpus_per_node)


def train_ep(args, model, train_loader, composite_attack, optimizer, criterion):
    global epoch
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # clean training
        if args.mode == 'natural':
            # zero gradient
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)

        # adv training normal
        elif args.mode == 'adv_train_madry':
            model.eval()
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = composite_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)

            model.train()

            # zero gradient
            optimizer.zero_grad()
            logits = model(data_adv)
            loss = criterion(logits, target)

        # adv training by trades
        elif args.mode == 'adv_train_trades':
            # TRADE Loss would require more memory.

            model.eval()
            batch_size = len(data)
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = composite_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)

            model.train()
            # zero gradient
            optimizer.zero_grad()

            # calculate robust loss
            logits = model(data)
            loss_natural = F.cross_entropy(logits, target)
            loss_robust = (1.0 / batch_size) * F.kl_div(F.log_softmax(model(data_adv), dim=1),
                                                        F.softmax(model(data), dim=1))
            loss = loss_natural + args.beta * loss_robust

        else:
            print("Not Specify Training Mode.")
            raise ValueError()

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                    100. * batch_idx / len(train_loader), loss.item()))


def train(model, optimizer, criterion, train_loader, train_sampler, test_loader, args, ngpus_per_node):
    global best_acc1, epoch, no_improve

    composite_attack = CompositeAttack(model, args.enable, mode='train', local_rank=args.rank, dataset='svhn',
                                       start_num=start_num, iter_num=iter_num,
                                       inner_iter_num=inner_iter_num, multiple_rand_start=True, order_schedule=args.order)

    for e in range(epoch, epoch + args.epochs):
        epoch = e
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, args)

        # adversarial training
        train_ep(args, model, train_loader, composite_attack, optimizer, criterion)

        # evaluation on natural examples
        test_loss, test_acc1 = eval_test(model, test_loader, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if is_best:
                no_improve = no_improve - (no_improve % 10)
            else:
                no_improve = no_improve + 1
            print("No improve: {}".format(no_improve))

            # save checkpoint
            print("Best Test Accuracy: {}%".format(best_acc1))

            filename = os.path.join(args.model_dir, 'model-epoch{}.pt'.format(e))
            torch.save({
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1,
                'no_improve': no_improve,
            }, filename)
            # print('Save model: {}'.format(os.path.join(args.model_dir, 'model-epoch{}.pt'.format(e))))
            if is_best:
                print("Save best model (epoch {})!".format(e))
                shutil.copyfile(filename, os.path.join(args.model_dir, 'model_best.pth'))
                print('Save model: {}'.format(os.path.join(args.model_dir, 'model_best.pth')))
            print('================================================================')
            with open(args.log_filename, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [e, test_loss, test_acc1, best_acc1]
                csv_write.writerow(data_row)


def eval_test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, args):
    """decrease the learning rate"""
    global epoch, no_improve
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
