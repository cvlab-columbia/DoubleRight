from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
# import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101, STL10, OxfordIIITPet, DTD

from torchvision.datasets import StanfordCars, Food101, SUN397, EuroSAT, \
    Caltech256, Country211, Flowers102, PCAM, FGVCAircraft  # , HatefulMemes

import torchvision.transforms as transforms
import torchvision
from dataloader import RandomLoader_L2

from clipfolder import clip
from models import prompters
from models.prompters import TokenPrompter
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, CLASS_SPLIT_CIFAR100
from utils import cosine_lr, convert_models_to_fp32, refine_classname

import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=3,
                        help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,  ## Why so large
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=5)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--model_arch', type=str, default='clip')
    parser.add_argument('--imagenet_root', type=str, default=None)

    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'null_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--CW', action='store_true')

    parser.add_argument('--pretrain_imgnet', action='store_true')
    parser.add_argument('--use_loss_normalize', action='store_true')

    parser.add_argument('--train_class_count', type=int, default=90)
    parser.add_argument('--noimginprop', action='store_true')

    args = parser.parse_args()

    args.filename = 'L2dr_dataset{}_{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.dataset, args.name, args.method, args.prompt_size, args.dataset,
               args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)

    return args


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(CIFAR100_STD).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu) / std


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X


# for multiGPU clip
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image, prompt_token=None, local_ind_prompt=None):
        return self.model.encode_image(image, prompt_token, local_ind_prompt=local_ind_prompt)

    ###


# alpha_test = 1. / 255
# attack_iters_test = 5
#
# epsilon = 2./255
upper_limit, lower_limit = 1, 0


def construct_prompt_full(des, is_SUN=False):
    all_att = []
    for each in des.keys():
        # ans.append(each)
        for com in des[each]:
            all_att.append(com.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower())
    all_att = list(set(all_att))

    ans = []
    for each in des.keys():
        # ans.append(each)
        for com in all_att:
            # ans.append(f'This is a photo of a {category}, which has {com}')
            if is_SUN:
                eacht = each.replace('_', ' ').replace('-', ' ').replace('/', '  ').lower()
            else:
                eacht = each.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower()
            comt = com.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower()
            ans.append(f'This is a photo of a {eacht} because there is {comt}')
    return ans

def construct_prompt_full_l2(des):
    all_att = []
    for each in des.keys():
        # ans.append(each)
        for com_dict in des[each]:
            com = list(com_dict.keys())[0]
            all_att.append(com.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower())
    all_att = list(set(all_att))

    all_att_l2 = []
    for each in des.keys():
        # ans.append(each)
        for com_dict in des[each]:
            com = list(com_dict.keys())[0]
            for l2atttmp in com_dict[com]:
                all_att_l2.append(l2atttmp.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower())
    all_att_l2 = list(set(all_att_l2))

    ans = []
    for each in des.keys():
        for att in all_att:
            for att_2 in all_att_l2:
                ans.append(f'This is a {att} of {each} because the {att} has {att_2}')
    return ans



def main():
    # name_test_list = ['food101_dr', 'caltech101', 'cifar10', 'cifar100']
    name_test_list = ['cifar10',  'cifar100', 'food101', 'caltech101', 'SUN']
    # name_test_list = ['cifar100']

    global best_acc1, device

    args = parse_option()
    args.use_loss_normalize = True

    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    import socket
    if socket.gethostname() == 'cv12' or socket.gethostname() == 'cv13':
        imagenet_root = '/local/ / /ImageNet-clean'
    else:
        imagenet_root = '/local/ /datasets/ImageNet-clean'

    if args.imagenet_root is not None:
        imagenet_root = args.imagenet_root

    # create model
    add_prompt_len = args.add_prompt_size

    model_names = ["RN50", "RN101", 'ViT-B/32', "ViT-B/16", "ViT-L/14"]

    model, preprocess = clip.load("ViT-L/14", device, jit=False, prompt_len=add_prompt_len)
    model_text, model_image = TextCLIP(model), ImageCLIP(model)

    convert_models_to_fp32(model_text)
    model_text = torch.nn.DataParallel(model_text)  # .to(device)
    model_text.eval()

    convert_models_to_fp32(model_image)
    model_image = torch.nn.DataParallel(model_image)  # .to(device)
    model_image.eval()

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)  # .to(device)
    model.eval()

    prompter = prompters.__dict__[args.method](args)  # .to(device)
    add_prompter = TokenPrompter(add_prompt_len, latentdim=1024)  # .to(device)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    # optionally resume from a checkpoint
    # assert args.resume is not None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            if 'vision_encoder_state_dict' in checkpoint.keys():
                # Load backbone for complementary experiment,
                # this assume that the finetuned model does not have the following prompts
                model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
                if 'image_encoder_dict' in checkpoint.keys():
                    model_image.module.load_state_dict(checkpoint['image_encoder_dict'], strict=False)
            else:
                # load only prompts, not backbone
                # if args.evaluate:
                    # if evaluate L2 prompt, then load
                prompter.load_state_dict(checkpoint['state_dict'])
                    # else, we are training this prompt for L2, and we always load addprompt for L1
                print('original', checkpoint['state_dict'])

                add_prompter.load_state_dict(checkpoint['add_prompter'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    # create data
    template = 'This is a photo of a {}'
    # print(f'template: {template}')

    # print(preprocess, 'preprocess')
    # exit()

    # TODO: we can train on cifar10 and test on cifar10, 100 in zero shot way, to see if generalize.
    preprocess = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])


    # DR validation list

    name_list = ['cifar10']
    root_path_list = ['/proj/vondrick4/ /DoubleRightDatasetOOD_split/cifar10_dr/test']
    json_list = ['/home/mcz/2022Fall/DoublyRight/GPT3/cifar10_text.json']

    root_L2_path_list = ['/proj/vondrick4/ /DoubleRightDatasetOOD_split/CIFAR10_L2_v4_dr/test']
    json_L2_list = ['/home/mcz/2022Fall/DoublyRight/GPT3/cifar10_level2_text.json']

    import json
    emb_attcat_list = []
    text_attcat_list = []
    l2emb_attcat_list = []
    l2text_attcat_list = []
    for name, root_path, json_path, l2root_path, l2json_path in zip(name_list, root_path_list, json_list, root_L2_path_list, json_L2_list):
        if name not in name_test_list:
            emb_attcat_list.append(None)
            text_attcat_list.append(None)
            continue

        with open(json_path, 'r') as f:
            des = json.load(f)
        texts = construct_prompt_full(des, name=='SUN')
        text_attcat_list.append(texts)
        block_size = 5000
        print(len(texts))
        if len(texts) < block_size:
            with autocast():
                with torch.no_grad():
                    cat_text_tokens = clip.tokenize(texts).to(device)
                    catatt_text_emb = model_text(cat_text_tokens)
                    catatt_text_emb = catatt_text_emb / catatt_text_emb.norm(dim=-1, keepdim=True)
        else:
            block_num = len(texts) // block_size + 1
            print('block num', block_num)
            catatt_text_emb = []
            with autocast():
                with torch.no_grad():
                    for ind in range(block_num):
                        cat_text_tokens_tmp = clip.tokenize(texts[ind * block_size:ind * block_size + block_size]).to(
                            device)
                        catatt_text_emb_tmp = model_text(cat_text_tokens_tmp)
                        catatt_text_emb_tmp = catatt_text_emb_tmp / catatt_text_emb_tmp.norm(dim=-1, keepdim=True)
                        catatt_text_emb.append(catatt_text_emb_tmp)

            catatt_text_emb = torch.cat(catatt_text_emb, dim=0)

        emb_attcat_list.append(catatt_text_emb)


        with open(l2json_path, 'r') as f:
            des2 = json.load(f)
        texts = construct_prompt_full_l2(des2)
        l2text_attcat_list.append(texts)
        block_size = 5000
        print(len(texts))
        if len(texts) < block_size:
            with autocast():
                with torch.no_grad():
                    cat_text_tokens = clip.tokenize(texts).to(device)
                    catatt_text_emb = model_text(cat_text_tokens)
                    l2catatt_text_emb = catatt_text_emb / catatt_text_emb.norm(dim=-1, keepdim=True)
        else:
            block_num = len(texts) // block_size + 1
            print('block num', block_num)
            catatt_text_emb = []
            with autocast():
                with torch.no_grad():
                    for ind in range(block_num):
                        cat_text_tokens_tmp = clip.tokenize(texts[ind * block_size:ind * block_size + block_size]).to(
                            device)
                        catatt_text_emb_tmp = model_text(cat_text_tokens_tmp)
                        catatt_text_emb_tmp = catatt_text_emb_tmp / catatt_text_emb_tmp.norm(dim=-1, keepdim=True)
                        catatt_text_emb.append(catatt_text_emb_tmp)

            l2catatt_text_emb = torch.cat(catatt_text_emb, dim=0)

        l2emb_attcat_list.append(l2catatt_text_emb)



    Dr_rootpath = '/proj/vondrick4/ /DoubleRightShared/preprocessed'
    Dr_rootpath = '/proj/vondrick4/ /DoubleRightDatasetOOD_split/cifar10_dr'
    args.Dr_rootpath = Dr_rootpath

    import socket
    if socket.gethostname() == 'cv12':
        imagenet_path = '/local/ / /ImageNet-clean/train'
    elif socket.gethostname() == 'cv11':
        imagenet_path = '/local/ /datasets/ImageNet-clean/train'
    else:
        imagenet_path = '/local/ /datasets/ImageNet-clean/train',



    train_dataset = RandomLoader_L2('/proj/vondrick4/ /DoubleRightDatasetOOD_split/CIFAR10_L2_v4_dr/train',
                                 composed_transforms=preprocess224_interpolate)
    train_json_path = '/home/mcz/2022Fall/DoublyRight/GPT3/cifar10_text.json'
    l2train_json_path = '/home/mcz/2022Fall/DoublyRight/GPT3/cifar10_level2_text.json'


    assert  args.dataset in name_test_list
    # for cid, e in enumerate(name_list):
    #     if e == args.dataset:

    # only train on cifar10 and test on cifar10
    train_text = text_attcat_list[0]
    train_emb_text = emb_attcat_list[0]

    l2train_text = l2text_attcat_list[0]
    l2train_emb_text = l2emb_attcat_list[0]
    # break

    # import pdb; pdb.set_trace()
    attcat2int = {}
    for ind, et in enumerate(train_text):
        attcat2int[et] = ind

    L2attcat2int = {}
    for ind, et in enumerate(l2train_text):
        L2attcat2int[et] = ind

    # import pdb; pdb.set_trace()

    train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=40, shuffle=True, sampler=None)


    # define criterion and optimizer
    optimizer = torch.optim.SGD(list(prompter.parameters()) + list(add_prompter.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_').replace('-', ' ').replace('/', ' ')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    dr_acc, dr_bl_acc = L2dr_validate(0, model_text, model_image, prompter, add_prompter, criterion, args,
                                      name_list, root_L2_path_list, l2emb_attcat_list,
                                      l2text_attcat_list, name_test_list)

    if args.evaluate:
        dr_acc, dr_bl_acc = dr_validate(0, model_text, model_image, prompter, add_prompter, criterion, args,
                                        name_list, root_path_list, json_list, emb_attcat_list,
                                        text_attcat_list, name_test_list
                                        )
        print('double right for level 1', dr_acc, dr_bl_acc)
        dr_acc, dr_bl_acc = L2dr_validate(0, model_text, model_image, prompter, add_prompter, criterion, args,
                                          name_list, root_L2_path_list, l2emb_attcat_list,
                                          l2text_attcat_list, name_test_list)

        print('double right level 2', dr_acc, dr_bl_acc)
        return

    epochs_since_improvement = 0

    is_best=True
    for epoch in range(args.epochs):

        train(train_loader, model, model_text, model_image, prompter, add_prompter,
              optimizer, scheduler,
              criterion, scaler, epoch, args, attcat2int, train_emb_text, L2attcat2int, l2train_emb_text)

        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            dr_acc, dr_bl_acc = dr_validate(0, model_text, model_image, prompter, add_prompter, criterion, args,
                                            name_list, root_path_list, json_list, emb_attcat_list,
                                            text_attcat_list, name_test_list
                                            )
            print('double right for level 1', dr_acc, dr_bl_acc)
            dr_acc, dr_bl_acc = L2dr_validate(epoch, model_text, model_image, prompter, add_prompter, criterion, args,
                                            name_list, root_L2_path_list, l2emb_attcat_list,
                                            l2text_attcat_list, name_test_list)

            print('double right level 2', dr_acc, dr_bl_acc)


        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    # wandb.run.finish()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


from utils import one_hot_embedding




def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)  # inside, already normalized and scaled.
    # img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)  # don't use this normalization during traiing, it destroys the model!!
    # scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True) # don't use this normalization during traiing, it destroys the model!!

    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text


# TODO: why in training, the acc for imagenet train is low, even after pretrained on imagenet.


def train(train_loader,  model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args, attcat2int, dr_text_emb, L2attcat2int, l2train_emb_text):
    """

    :param train_loader:
    :param attributes_all:
    :param categories_all:
    :param model:
    :param model_text:
    :param model_image:
    :param prompter:
    :param add_prompter:
    :param optimizer:
    :param scheduler:
    :param criterion:
    :param scaler:
    :param epoch:
    :param args:
    :param cat2att: dictionary, given a category, give a list of the attribute list.
    :param catatt_text_emb: the embedding vector, precomputed, for category+attribute language embedding.
    :param text_cat_startindex: dictionary, indicate for each category, where is the first vector of that category start
    :return:
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    L2top1 = AverageMeter('Acc@1', ':6.2f')
    # imgnet_top1 = AverageMeter('ImgNet Acc@1', ':6.2f')
    # dr_classification_top1 = AverageMeter('DR ImgNet Acc@1', ':6.2f')
    # img_classification_top1 = AverageMeter('Img ImgNet Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, L2top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()
    add_prompter.train()

    num_batches_per_epoch = len(train_loader)

    # import pdb; pdb.set_trace() # text_cat_endindex   KeyError: 'amphibian'

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    # print('text token', texts)

    end = time.time()

    # TODO: may save the big text in advance? Too big to save?
    # Going to show improve interpretability, and improve in ImageNet accuracy, and maybe its ood.
    for i, (images, cate_name, att_name, L2_att_name) in enumerate(tqdm(train_loader)):
        bs = len(cate_name)
        # direct tag each cat+att with an integer from the groundtruth list.
        # This may not work for imagenet.
        target = []
        # for level 1
        # import pdb; pdb.set_trace()
        for cat, att in zip(cate_name, att_name):
            prompt_why = f'This is a photo of a {cat} because there is {att}'
            if prompt_why in attcat2int:
                target.append(attcat2int[prompt_why])
            else:
                for each in attcat2int.keys():
                    if cat in each:  # last bug ' or ' relace /
                        print(each)
                import pdb;
                pdb.set_trace()

        l2target = []
        for cat, att, att_2 in zip(cate_name, att_name, L2_att_name):
            prompt_why = f'This is a {att} of {cat} because the {att} has {att_2}'
            if prompt_why in L2attcat2int:
                l2target.append(L2attcat2int[prompt_why])
            else:
                for each in L2attcat2int.keys():
                    if cat in each:  # last bug ' or ' relace /
                        print(each)
                import pdb;
                pdb.set_trace()


        target = torch.LongTensor(target).to(device)
        l2target = torch.LongTensor(l2target).to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)

        # K = 20
        # imgnet_text = create_train_text_prompts_imagenet(pseudo_gt, englishname, attributes_all, categories_all, K=K)
        # imgnet_text_tokens = clip.tokenize(imgnet_text).to(device)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()
        K = 50

        with autocast():
            logit_scale = model.module.logit_scale.exp()

            prompted_images = prompter(clip_img_preprocessing(images))  # prompter(images)
            prompt_token = add_prompter()

            if prompt_token is not None:
                bs = images.size(0)
                prompt_token = prompt_token.repeat(bs, 1, 1)

            drimg_fea = model_image(prompted_images, prompt_token=prompt_token)


            drimg_fea = logit_scale * drimg_fea / drimg_fea.norm(dim=1, keepdim=True)

            logit_dr = drimg_fea @ dr_text_emb.t()


            L2logit_dr = drimg_fea @ l2train_emb_text.t()


            loss1 = criterion(logit_dr, target)
            loss2 = criterion(L2logit_dr, l2target)

            # loss = loss1 + loss2 # + loss3
            loss = loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(logit_dr, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        L2acc1 = accuracy(L2logit_dr, l2target, topk=(1,))
        L2top1.update(L2acc1[0].item(), images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break
            # break

            # if args.use_wandb:
            #     wandb.log({
            #         'training_loss': losses.avg,
            #         'training_acc': top1.avg
            #          })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


# TODO: top k category, then use them for attributes
# TODO: precompute all text embedding.

def L2dr_validate(epoch, model_text, model_image, prompter, add_prompter, criterion, args, name_list, root_path_list, emb_attcat_list,
                text_attcat_list, name_test_list):

    for name, root_path, catatt_text_emb, texts in zip(name_list, root_path_list, emb_attcat_list, text_attcat_list):

        if name not in name_test_list:
            continue
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        RR_Whole_score = 0
        RW_Whole_score = 0
        WR_Whole_score = 0
        WW_Whole_score = 0
        cat_correct_whole = 0
        Whole_cnt = 0

        bl_Whole_score = 0

        for category_ori in os.listdir(root_path):
            num_of_attribute = len(os.listdir(os.path.join(root_path, category_ori)))
            for ccant, attribute in enumerate(os.listdir(os.path.join(root_path, category_ori))):

                for subcnt, l2attribute in enumerate(os.listdir(os.path.join(root_path, category_ori, attribute))):

                    # Test if attribute is only partial, where another half in subdir due to the /
                    att_dir = os.path.join(root_path, category_ori, attribute, l2attribute)

                    # ==========================================================
                    correct = False
                    for each in os.listdir(att_dir):
                        if '0' == each or '1' == each:
                            # print('correct dir')
                            correct = True
                            break
                    if not correct:
                        # subdir is also attribute
                        assert len(os.listdir(att_dir)) == 1
                        att_dir = os.path.join(att_dir, os.listdir(att_dir)[0])
                        attribute = attribute + '/' + os.listdir(att_dir)[0]
                        # generate the right path and the right attribute name for it.

                    # do this again for cases where multiple / exists
                    correct = False
                    for each in os.listdir(att_dir):
                        if '0' == each or '1' == each:
                            # print('correct dir')
                            correct = True
                            break
                    if not correct:
                        # subdir is also attribute
                        assert len(os.listdir(att_dir)) == 1
                        att_dir = os.path.join(att_dir, os.listdir(att_dir)[0])
                        attribute = attribute + '/' + os.listdir(att_dir)[0]
                    # ==========================================================


                    val_dataset = torchvision.datasets.ImageFolder(
                        att_dir,
                        transform=preprocess
                    )
                    train_sampler = None
                    val_sampler = None

                    # try:
                    val_loader = DataLoader(val_dataset,
                                            batch_size=20, pin_memory=True,
                                            num_workers=1, shuffle=False, sampler=train_sampler)

                    RR_Score_cnt=0
                    RW_Score_cnt = 0
                    WR_Score_cnt = 0
                    WW_Score_cnt = 0

                    catonly_acc=0
                    cnt=0
                    category = category_ori.replace('_', ' ')

                    cat_correct = 0
                    for i, (images, target) in enumerate(val_loader):
                        images = images.to(device)
                        cnt += images.size(0)
                        # target = target.to(device)

                        with autocast():
                            # clean images, with prompt and without prompt
                            # compute output
                            with torch.no_grad():
                                # image_fea, text_fea = model(images, text_tokens, return_fea=True)
                                prompt_token = add_prompter()
                                if prompt_token is not None:
                                    bs = images.size(0)
                                    prompt_token = prompt_token.repeat(bs, 1, 1)


                                image_fea = model_image(prompter(clip_img_preprocessing(images)), prompt_token=prompt_token)
                                image_fea /= image_fea.norm(dim=-1, keepdim=True)

                                logits_per_image = image_fea @ catatt_text_emb.t()
                                _, topk_cat_indices = torch.topk(logits_per_image, 5, dim=1)


                                attribute = attribute.replace('__', '/')  # convert back to json's format.
                                # print("topk_cat_indices", topk_cat_indices, cat_index)

                                for bb in range(topk_cat_indices.size(0)): # for each intance in batch
                                    tops = []
                                    category_histogram={}
                                    reason_right = False
                                    together_right = False
                                    correct_category_bool = False

                                    # cate_text = cate_prompt[topk_catonly_indices[bb,0].item()] # for debug
                                    # if category in cate_text:
                                    #     catonly_acc += 1

                                    together_reason_right = False
                                    for ll in range(topk_cat_indices.size(1)):  # for each top k
                                        tops.append(texts[topk_cat_indices[bb, ll].item()])

                                        # f'This is a {att} of {each} because the {att} has {att_2}'

                                        tmp_text = texts[topk_cat_indices[bb, ll].item()]
                                        cat_, att_all = tmp_text.split(' because the ')
                                        cat_ = cat_.replace('This is a ', '')

                                        splitlist = cat_.split(' of ')
                                        if len(splitlist) == 2:
                                            att, cat_ = splitlist
                                        else:
                                            cat_ = splitlist[-1]
                                            att=""
                                            for fmg in splitlist[:-1]:
                                                if att=="":
                                                    att=fmg
                                                else:
                                                    att = att+" of "+fmg

                                            # import pdb;
                                            # pdb.set_trace()

                                            att_L2 = att_all.split(' has ')[1]


                                        # print()

                                        if cat_ in category_histogram:
                                            category_histogram[cat_] += 1
                                        else:
                                            category_histogram[cat_] = 1

                                        if category in texts[topk_cat_indices[bb, ll].item()] and attribute in texts[topk_cat_indices[bb, ll].item()] and l2attribute in texts[topk_cat_indices[bb, ll].item()]:
                                            # if get both category and attribute correct
                                            # find specified category in top k reterival
                                            # RR_Score_cnt += 1
                                            together_reason_right = True
                                            break

                                        if attribute in texts[topk_cat_indices[bb, ll].item()]:
                                            reason_right = True

                                    # print('\n\ngroundtruth', category_ori, attribute, cate_text)
                                    # print(tops)

                                    prediction=None
                                    maxcnt=-1
                                    for key_value in category_histogram.keys():
                                        if category_histogram[key_value] > maxcnt:
                                            prediction = key_value
                                            maxcnt = category_histogram[key_value]

                                    # Top k based accuracy is right
                                    if prediction == category_ori:
                                        cat_correct += 1

                                    if prediction == category_ori and together_reason_right: # right reason help get the right prediction
                                        RR_Score_cnt += 1
                                    elif prediction == category_ori: # prediction right, but none reason is right
                                        RW_Score_cnt += 1

                                    # prediction wrong
                                    if prediction != category_ori:
                                        wr = False
                                        for ll in range(topk_cat_indices.size(1)):  # for each top k
                                            # still get the right reason that exist in the photo
                                            if prediction in texts[topk_cat_indices[bb, ll].item()] and attribute in texts[
                                                topk_cat_indices[bb, ll].item()] and l2attribute in texts[topk_cat_indices[bb, ll].item()]:

                                                WR_Score_cnt += 1 # wrong using the right reason
                                                wr = True  # flag for wrong using the right reason
                                                break
                                        # does not get any right reason
                                        if wr is False:
                                            WW_Score_cnt += 1


                        # print('\n\n\n\nresult', category +'-' + attribute, RR_Score_cnt * 1.0 / cnt, 'RW',
                        #       RW_Score_cnt / cnt, 'WR', WR_Score_cnt / cnt, 'WW', WW_Score_cnt / cnt)

                        RR_Whole_score += RR_Score_cnt
                        RW_Whole_score += RW_Score_cnt
                        WR_Whole_score += WR_Score_cnt
                        WW_Whole_score += WW_Score_cnt
                        cat_correct_whole += cat_correct
                        Whole_cnt += cnt

                    # print(name, 'total score', RR_Whole_score / Whole_cnt, RW_Whole_score / Whole_cnt, WR_Whole_score / Whole_cnt,
                    #       WW_Whole_score / Whole_cnt, 'Classification Acc', cat_correct_whole/Whole_cnt)

                    # except:
                    #     continue

                # except:
                #     continue


        # print(RR_Score_dict)

        print(epoch, name, 'total score', RR_Whole_score / Whole_cnt, RW_Whole_score/Whole_cnt, WR_Whole_score/Whole_cnt,
              WW_Whole_score/Whole_cnt, cat_correct_whole/Whole_cnt)


    return 0,0


def dr_validate(epoch, model_text, model_image, prompter, add_prompter, criterion, args, name_list, root_path_list, json_list, emb_attcat_list,
                text_attcat_list, name_test_list):

    for name, root_path, json_path, catatt_text_emb, texts in zip(name_list, root_path_list, json_list, emb_attcat_list, text_attcat_list):

        if name not in name_test_list:
            continue
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        RR_Whole_score = 0
        RW_Whole_score = 0
        WR_Whole_score = 0
        WW_Whole_score = 0
        cat_correct_whole = 0
        Whole_cnt = 0

        bl_Whole_score = 0

        for category_ori in os.listdir(root_path):
            num_of_attribute = len(os.listdir(os.path.join(root_path, category_ori)))
            for ccant, attribute in enumerate(os.listdir(os.path.join(root_path, category_ori))):
                # Test if attribute is only partial, where another half in subdir due to the /
                att_dir = os.path.join(root_path, category_ori, attribute)

                # ==========================================================
                correct = False
                for each in os.listdir(att_dir):
                    if '0' == each or '1' == each:
                        # print('correct dir')
                        correct = True
                        break
                if not correct:
                    # subdir is also attribute
                    assert len(os.listdir(att_dir)) == 1
                    att_dir = os.path.join(att_dir, os.listdir(att_dir)[0])
                    attribute = attribute + '/' + os.listdir(att_dir)[0]
                    # generate the right path and the right attribute name for it.

                # do this again for cases where multiple / exists
                correct = False
                for each in os.listdir(att_dir):
                    if '0' == each or '1' == each:
                        # print('correct dir')
                        correct = True
                        break
                if not correct:
                    # subdir is also attribute
                    assert len(os.listdir(att_dir)) == 1
                    att_dir = os.path.join(att_dir, os.listdir(att_dir)[0])
                    attribute = attribute + '/' + os.listdir(att_dir)[0]
                # ==========================================================


                val_dataset = torchvision.datasets.ImageFolder(
                    att_dir,
                    transform=preprocess
                )
                train_sampler = None
                val_sampler = None

                # try:
                val_loader = DataLoader(val_dataset,
                                        batch_size=20, pin_memory=True,
                                        num_workers=1, shuffle=False, sampler=train_sampler)

                RR_Score_cnt=0
                RW_Score_cnt = 0
                WR_Score_cnt = 0
                WW_Score_cnt = 0

                catonly_acc=0
                cnt=0
                category = category_ori.replace('_', ' ')

                cat_correct = 0
                for i, (images, target) in enumerate(val_loader):
                    images = images.to(device)
                    cnt += images.size(0)
                    # target = target.to(device)

                    with autocast():
                        # clean images, with prompt and without prompt
                        # compute output
                        with torch.no_grad():
                            # image_fea, text_fea = model(images, text_tokens, return_fea=True)
                            prompt_token = add_prompter()
                            if prompt_token is not None:
                                bs = images.size(0)
                                prompt_token = prompt_token.repeat(bs, 1, 1)


                            image_fea = model_image(prompter(clip_img_preprocessing(images)), prompt_token=prompt_token)
                            image_fea /= image_fea.norm(dim=-1, keepdim=True)

                            logits_per_image = image_fea @ catatt_text_emb.t()
                            _, topk_cat_indices = torch.topk(logits_per_image, 5, dim=1)


                            attribute = attribute.replace('__', '/')  # convert back to json's format.
                            # print("topk_cat_indices", topk_cat_indices, cat_index)

                            for bb in range(topk_cat_indices.size(0)): # for each intance in batch
                                tops = []
                                category_histogram={}
                                reason_right = False
                                together_right = False
                                correct_category_bool = False

                                # cate_text = cate_prompt[topk_catonly_indices[bb,0].item()] # for debug
                                # if category in cate_text:
                                #     catonly_acc += 1

                                together_reason_right = False
                                for ll in range(topk_cat_indices.size(1)):  # for each top k
                                    tops.append(texts[topk_cat_indices[bb, ll].item()])

                                    tmp_text = texts[topk_cat_indices[bb, ll].item()]


                                    cat_, att = tmp_text.split(' because there is ')
                                    cat_ = cat_.replace('This is a photo of a ', '')
                                    if cat_ in category_histogram:
                                        category_histogram[cat_] += 1
                                    else:
                                        category_histogram[cat_] = 1

                                    if category in texts[topk_cat_indices[bb, ll].item()] and attribute in texts[topk_cat_indices[bb, ll].item()]:
                                        # if get both category and attribute correct
                                        # find specified category in top k reterival
                                        # RR_Score_cnt += 1
                                        together_reason_right = True
                                        break

                                    if attribute in texts[topk_cat_indices[bb, ll].item()]:
                                        reason_right = True

                                # print('\n\ngroundtruth', category_ori, attribute, cate_text)
                                # print(tops)

                                prediction=None
                                maxcnt=-1
                                for key_value in category_histogram.keys():
                                    if category_histogram[key_value] > maxcnt:
                                        prediction = key_value
                                        maxcnt = category_histogram[key_value]

                                # Top k based accuracy is right
                                if prediction == category_ori:
                                    cat_correct += 1

                                if prediction == category_ori and together_reason_right: # right reason help get the right prediction
                                    RR_Score_cnt += 1
                                elif prediction == category_ori: # prediction right, but none reason is right
                                    RW_Score_cnt += 1

                                # prediction wrong
                                if prediction != category_ori:
                                    wr = False
                                    for ll in range(topk_cat_indices.size(1)):  # for each top k
                                        # still get the right reason that exist in the photo
                                        if prediction in texts[topk_cat_indices[bb, ll].item()] and attribute in texts[
                                            topk_cat_indices[bb, ll].item()]:

                                            WR_Score_cnt += 1 # wrong using the right reason
                                            wr = True  # flag for wrong using the right reason
                                            break
                                    # does not get any right reason
                                    if wr is False:
                                        WW_Score_cnt += 1


                    # print('\n\n\n\nresult', category +'-' + attribute, RR_Score_cnt * 1.0 / cnt, 'RW',
                    #       RW_Score_cnt / cnt, 'WR', WR_Score_cnt / cnt, 'WW', WW_Score_cnt / cnt)

                    RR_Whole_score += RR_Score_cnt
                    RW_Whole_score += RW_Score_cnt
                    WR_Whole_score += WR_Score_cnt
                    WW_Whole_score += WW_Score_cnt
                    cat_correct_whole += cat_correct
                    Whole_cnt += cnt

                    # print(name, 'total score', RR_Whole_score / Whole_cnt, RW_Whole_score / Whole_cnt, WR_Whole_score / Whole_cnt,
                    #       WW_Whole_score / Whole_cnt, 'Classification Acc', cat_correct_whole/Whole_cnt)

                    # except:
                    #     continue

                # except:
                #     continue


        # print(RR_Score_dict)

        print(epoch, name, 'total score', RR_Whole_score / Whole_cnt, RW_Whole_score/Whole_cnt, WR_Whole_score/Whole_cnt,
              WW_Whole_score/Whole_cnt, cat_correct_whole/Whole_cnt)


    return 0,0


# def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
def validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
             prompter, add_prompter, criterion, args):
    dataset_num = len(val_loader_list)
    acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        # print('text', texts)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()

        # print(val_dataset_name, 'text token', texts_list)

        #
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            # if 'cifar' not in val_dataset_name:
            #     if i % 20 != 0 and not args.evaluate:
            #         continue

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # print(images.size())

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    prompt_token = add_prompter()
                    # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                    output_prompt, _ = multiGPU_CLIP(model_image, model_text, model,
                                                     prompter(clip_img_preprocessing(images)), text_tokens,
                                                     prompt_token)

                    # output_org, _ = model(clip_img_preprocessing(images), text_tokens)
                    output_org, _ = multiGPU_CLIP(model_image, model_text, model, clip_img_preprocessing(images),
                                                  text_tokens, None)
                    #

                    # loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    # losses.update(loss.item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.debug:
                    break

        torch.cuda.empty_cache()

        print(dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
                      top1_prompt=top1_prompt, top1_org=top1_org))
        acc_all.append(top1_adv_prompt.avg)

    # if args.use_wandb:
    #     wandb.log({
    #         'val_loss': losses.avg,
    #         'val_acc_prompt': top1_prompt.avg,
    #         'val_acc_org': top1_org.avg,
    #     })

    return np.mean(acc_all)


if __name__ == '__main__':
    main()
