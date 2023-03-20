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
    Caltech256, Country211, Flowers102, PCAM, FGVCAircraft #, HatefulMemes

import torchvision.transforms as transforms
import torchvision
from dataloader import RandomLoader_combined

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
    parser.add_argument('--batch_size', type=int, default=256,
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

    args.filename = 'dr_mix_pt{}_lossn{}_{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.pretrain_imgnet, args.use_loss_normalize, args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
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


def main():
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
        imagenet_root = '/local/ImageNet-clean'
    else:
        imagenet_root = '/local/ImageNet-clean'

    if args.imagenet_root is not None:
        imagenet_root = args.imagenet_root

    # create model
    add_prompt_len = args.add_prompt_size

    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
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
    add_prompter = TokenPrompter(add_prompt_len, latentdim=768)  # .to(device)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    # optionally resume from a checkpoint
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
                prompter.load_state_dict(checkpoint['state_dict'])
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

    if args.dataset == 'cifar100':

        train_dataset = CIFAR100(args.root, transform=preprocess,
                                 download=True, train=True)

        val_dataset = CIFAR100(args.root, transform=preprocess,
                               download=True, train=False)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                              download=True, train=False)

    elif args.dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(imagenet_root, 'train'),
            transform=preprocess224
        )

    elif args.dataset == 'zeroshot_cifar100':
        train_class_count = args.train_class_count
        train_dataset = CLASS_SPLIT_CIFAR100(args.root, transform=preprocess, download=True,
                                             train=True, train_class_count=train_class_count, load_train_classes=True)

    val_dataset_list = []

    if args.evaluate:
        val_dataset_name = ['cifar10', 'cifar100', 'STL10', 'SUN397', 'StanfordCars', 'Food101',
                            'oxfordpet', 'flowers102', 'Country211', 'dtd', 'EuroSAT', 'fgvc_aircraft',
                            'PCAM', 'hateful_memes',
                            'ImageNet', 'Caltech101', 'Caltech256']
        # val_dataset_name = ['PCAM', 'hateful_memes',
        #                     'ImageNet', 'Caltech101', 'Caltech256']
    else:
        val_dataset_name = ['cifar10', 'cifar100', 'dtd', 'EuroSAT', ]

    # val_dataset_name = ['Caltech101', 'PCAM', 'hateful_memes', 'ImageNet', 'cifar10'] # missing for previous evaluation

    #  'Caltech256', has problem


    val_dataset_name = ['ImageNet', 'ImageNetV2']
    # val_dataset_name = ['ImageNet']



    for each in val_dataset_name:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess,
                                            download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess,
                                             download=True, train=False))
        elif each == 'zeroshot_cifar100':
            assert args.dataset == 'zeroshot_cifar100'
            val_dataset_list.append(CLASS_SPLIT_CIFAR100(args.root, transform=preprocess, download=True,
                                                         train=False, train_class_count=train_class_count,
                                                         load_train_classes=False))

        elif each == 'zeroshot_cifar100_overlap':
            assert args.dataset == 'zeroshot_cifar100'
            val_dataset_list.append(CLASS_SPLIT_CIFAR100(args.root, transform=preprocess, download=True,
                                                         train=False, train_class_count=train_class_count,
                                                         load_train_classes=True))

        elif each == 'Caltech101':
            val_dataset_list.append(Caltech101(args.root, target_type='category', transform=preprocess224,
                                               download=True))
        elif each == 'PCAM':
            val_dataset_list.append(PCAM(args.root, split='test', transform=preprocess224,
                                         download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test',
                                          transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root,
                                           transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test',
                                                 transform=preprocess224, download=True))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test',
                                            transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test',
                                                  transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(EuroSAT(args.root,
                                            transform=preprocess224, download=True))

        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224,
                                               download=True))
        # elif each == 'FER2013':
        #     val_dataset_list.append(OxfordIIITPet(args.root, split='test',
        #                                           transform=preprocess224, download=True))
        elif each == 'flowers102':
            val_dataset_list.append(Flowers102(args.root, split='test',
                                               transform=preprocess224, download=True))
        elif each == 'Country211':
            val_dataset_list.append(Country211(args.root, split='test',
                                               transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(DTD(args.root, split='test',
                                        transform=preprocess224, download=True))

        elif each == 'fgvc_aircraft':
            val_dataset_list.append(FGVCAircraft(args.root, split='test',
                                                 transform=preprocess224, download=True))
        elif each == 'hateful_memes':
            val_dataset_list.append(HatefulMemes(args.root, splits=['test_seen', 'test_unseen'],
                                                 transform=preprocess224_interpolate))

        elif each == 'ImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(imagenet_root, 'val'),
                transform=preprocess224))

        elif each == 'ImageNetV2':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                '/proj/ImageNetV2',
                transform=preprocess224))

            # val_dataset_list.append(torchvision.datasets.ImageNet(
            # root=imagenet_root,
            # split='val',
            # transform=preprocess224))

    train_sampler = None
    val_sampler = None

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch_size, pin_memory=True,
    #                           num_workers=args.num_workers, shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each,
                                  batch_size=args.batch_size*2, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler) for each in
                       val_dataset_list]

    class_names = train_dataset.classes

    Dr_rootpath = '/proj/DoubleRightShared/preprocessed'
    args.Dr_rootpath = Dr_rootpath

    import socket
    if socket.gethostname() == 'cv12':
        imagenet_path = '/local/ImageNet-clean/train'
    elif socket.gethostname() == 'cv11':
        imagenet_path = '/local/ImageNet-clean/train'
    else:
        imagenet_path = '/local/ImageNet-clean/train'

    train_dataset = RandomLoader_combined(f'{Dr_rootpath}/train', imagenet_path=imagenet_path,
                                 composed_transforms=preprocess224_interpolate)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=40, shuffle=True, sampler=None)

    train_imagenet_dataset = torchvision.datasets.ImageFolder(
        imagenet_path,
        transform=preprocess224
        )
    train_imagenet_loader = DataLoader(train_imagenet_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=40, shuffle=True, sampler=None)



    if args.dataset == 'ImageNet' or args.dataset == 'ImageNetV2':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    # texts_train = [template.format(label) for label in class_names]

    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts
        else:
            class_names = each.classes
            if val_dataset_name[cnt] == 'ImageNet' or val_dataset_name[cnt] == 'ImageNetV2':
                from utils import load_imagenet_folder2name
                folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name].replace('_', ' ').replace('-', ' ').replace('/', ' '))
                class_names = new_class_names
                # class_names=class_names
            # else:
            #     print('val_dataset_name[cnt]', val_dataset_name[cnt])
            #     class_names = each.classes


            class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    print(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)

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

    import json
    with open('Our_ImageNet_attribute.json', 'r') as f:
        des = json.load(f)

    def get_attributes(des):
        cat = []
        ans = []
        for each in des.keys():
            # ans.append(each)
            cat.append(each)
            for com in des[each]:
                ans.append(com)
        return ans, cat

    def get_category_attributes_map(des):
        cat2att = {}
        for each in des.keys():
            # ans.append(each)
            cat2att[each.replace('_', ' ').replace('-', ' ').replace('/', ' ')] = []
            for com in des[each]:
                cat2att[each.replace('_', ' ').replace('-', ' ').replace('/', ' ')].append(com)
        return cat2att

    all_attribute, all_caten = get_attributes(des)
    cat2att = get_category_attributes_map(des)

    # make dir
    refined_template = template.lower().replace(' ', '_').replace('-', ' ').replace('/', ' ')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    # if args.use_wandb:
    #     wandb.init(project='Visual Prompting')
    #     wandb.config.update(args)
    #     wandb.run.name = args.filename
    #     wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                             prompter, add_prompter, criterion, args)
        dr_acc, dr_bl_acc = dr_validate(model, prompter, add_prompter, criterion, args)
        print('double right', dr_acc, dr_bl_acc)
        return

    epochs_since_improvement = 0

    # get embedding for imagenet list
    text_prompts = []
    text_cat_endindex = {}
    cnt_index = 0
    # import pdb; pdb.set_trace()
    for eachcategory in cat2att.keys():
        start_ind = cnt_index
        for each_att in cat2att[eachcategory]:
            cnt_index += 1
            text_prompts.append('This is a photo of a {} because there is {}'.format(eachcategory.replace('_', ' ').replace('-', ' ').replace('/', ' '), each_att))
        text_cat_endindex[eachcategory.replace('_', ' ').replace('-', ' ').replace('/', ' ')] = (start_ind, cnt_index)

    text_prompts_tokens = clip.tokenize(text_prompts).to(device)

    texts = [template.format(label.replace('_', ' ').replace('-', ' ').replace('/', ' ')) for label in class_names]
    texts_tokens = clip.tokenize(texts).to(device)

    with autocast():
        with torch.no_grad():
            catatt_text_emb = model_text(text_prompts_tokens)
            cat_text_emb = model_text(texts_tokens)
            if args.use_loss_normalize:
                catatt_text_emb = catatt_text_emb / catatt_text_emb.norm(dim=1, keepdim=True)
                cat_text_emb = cat_text_emb / cat_text_emb.norm(dim=1, keepdim=True)

    # =====================================
    class_names_map={}
    for itr, na in enumerate(class_names):
        class_names_map[na.replace('_', ' ').replace('-', ' ').replace('/', ' ')] = itr


    # for ee in range(10):
    #     train_ori(train_imagenet_loader, cat_text_emb, model, model_text, model_image, prompter, add_prompter,
    #               optimizer, scheduler, criterion, scaler, ee, args)
    # #
    # acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
    #                      prompter, add_prompter, criterion, args)

    # imgtrain = torchvision.datasets.ImageFolder(
    #     os.path.join(imagenet_root, 'train'),
    #     transform=preprocess224)
    #
    # imgtrainloader = DataLoader(imgtrain,
    #            batch_size=args.batch_size * 2, pin_memory=True,
    #            num_workers=args.num_workers, shuffle=False, sampler=val_sampler)
    #
    # acc1_mean = validate([imgtrainloader], val_dataset_name, texts_list, model, model_text, model_image,
    #                      prompter, add_prompter, criterion, args)

    for epoch in range(args.epochs):
        #
        if args.pretrain_imgnet:
            train_ori(train_imagenet_loader, cat_text_emb, model, model_text, model_image, prompter, add_prompter,
                  optimizer, scheduler, criterion, scaler, epoch, args)
        else:
            train(train_loader, all_attribute, all_caten, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler,
                  criterion, scaler, epoch, args, cat2att, catatt_text_emb, text_cat_endindex, text_prompts, cat_text_emb, class_names_map)

        # evaluate on validation set
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                             prompter, add_prompter, criterion, args)
        if epoch % args.validate_freq == 5:
            dr_acc, dr_bl_acc = dr_validate(model, prompter, add_prompter, criterion, args)
            print('double right', dr_acc, dr_bl_acc)

        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

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


def attack_CW(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
              attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_CW_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _ = multiGPU_CLIP(model_image, model_text, model, _images, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class)
        label_mask = label_mask.cuda()

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()

        output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _ = multiGPU_CLIP(model_image, model_text, model, _images, text_tokens, None)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token) # inside, already normalized and scaled.
    # img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)  # don't use this normalization during traiing, it destroys the model!!
    # scale_text_embed = scale_text_embed / scale_text_embed.norm(dim=-1, keepdim=True) # don't use this normalization during traiing, it destroys the model!!

    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text


def train_ori(train_loader, cat_emb, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    imgnet_top1 = AverageMeter('ImgNet Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, imgnet_top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()
    add_prompter.train()

    num_batches_per_epoch = len(train_loader)

    for i, (images, target) in enumerate(tqdm(train_loader)):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)

        with autocast():
            prompt_token = add_prompter()
            if prompt_token is not None:
                bs = images.size(0)
                prompt_token = prompt_token.repeat(bs, 1, 1)
            image_fea = model_image(prompter(clip_img_preprocessing(images)), prompt_token=prompt_token, local_ind_prompt=None)

            if args.use_loss_normalize:
                image_fea = image_fea / image_fea.norm(dim=1, keepdim=True)
                logit_scale = model.module.logit_scale.exp()
                image_fea = logit_scale * image_fea

            cat_emb_tmp = cat_emb

            cat_logit_per_image = image_fea @ cat_emb_tmp.t()

            loss = criterion(cat_logit_per_image, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()
        # model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(cat_logit_per_image, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        if i % args.print_freq == 0:
            progress.display(i)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': prompter.state_dict(),
        'add_prompter': add_prompter.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, args)

    return losses.avg, top1.avg


# TODO: why in training, the acc for imagenet train is low, even after pretrained on imagenet.


def train(train_loader, attributes_all, categories_all, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args, cat2att, catatt_text_emb, text_cat_endindex,
          text_prompts, cate_embedd, class_names):
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
    imgnet_top1 = AverageMeter('ImgNet Acc@1', ':6.2f')
    dr_classification_top1 = AverageMeter('DR ImgNet Acc@1', ':6.2f')
    img_classification_top1 = AverageMeter('Img ImgNet Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, imgnet_top1, dr_classification_top1, img_classification_top1],
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
    for i, (images, cate_name, att_name, imgnetimg, englishname) in enumerate(tqdm(train_loader)):
        # print('image', images.max(), images.min())
        # print('img', cate_name, att_name, 'imgnet', englishname)

        class_target = []
        # import pdb; pdb.set_trace()
        for each_dr in cate_name:
            class_target.append(class_names[each_dr])
        for each_dr in englishname:
            class_target.append(class_names[each_dr])



        def create_train_text_prompts(cate_name, att_name, attributes_all, categories_all, K=1000): # two parts, one make sure model prediction is right, another part make sure interpretation is right.
            texts = []
            # groundtruth
            for cat, att in zip(cate_name, att_name):
                template = 'This is a photo of a {} because there is {}'.format(cat, att)
                texts.append(template)

            # cateogry distractors
            remains = list(set(categories_all) - set(cate_name))
            # print('remaining cat', len(remains))

            for each in remains:
                texts.append('This is a photo of a {}'.format(each))

            # reason distractors
            for eachcat, att in zip(cate_name, att_name):
                # randomly sample
                remain_att = list(set(attributes_all) - set([att]))

                select_neg_att = random.sample(remain_att, K)

                for e_select_att in select_neg_att:
                    texts.append('This is a photo of a {} because there is {}'.format(eachcat, e_select_att))
            return texts

        def create_train_text_prompts_imagenet(pseudo_gt, englishname_gt, attributes_all, categories_all, K=10):
            # add distractors, category
            remain_category = list(set(categories_all) - set(englishname_gt))
            # import pdb; pdb.set_trace()
            ans = []
            for e in remain_category:
                ans.append(f'This is a photo of a {e}')

            # add distractors, reason
            for e in pseudo_gt:
                e = e[21:]
                # print(e, 'bingo')
                this_category, this_attribute = e.split(' because there is ')

                # import pdb; pdb.set_trace()
                pos_atts = cat2att[this_category]
                att_remains = list(set(attributes_all) - set(pos_atts))
                select_neg_att = random.sample(att_remains, K)
                
                for e_select_att in select_neg_att:
                    ans.append('This is a photo of a {} because there is {}'.format(this_category, e_select_att))

            return pseudo_gt + ans


        bs = len(cate_name)
        target = torch.arange(bs).cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # print('bs', BATCH_SIZE)
        images = images.to(device)
        imgnetimg = imgnetimg.to(device)
        target = target.to(device)

        cat_target = torch.LongTensor(class_target).to(device)

        # getting pseudo label for attribute
        pseudo_gt = []
        with autocast():
            with torch.no_grad():

                image_fea = model_image(clip_img_preprocessing(imgnetimg), prompt_token=None, local_ind_prompt=None)
                if args.use_loss_normalize:
                    image_fea = image_fea / image_fea.norm(dim=1, keepdim=True)
                    logit_scale = model.module.logit_scale.exp()
                    image_fea = logit_scale * image_fea

                cat_logit_per_image = image_fea @ catatt_text_emb.t()
                mask = torch.zeros_like(cat_logit_per_image)
                for tmpcnt, eachcat in enumerate(englishname):
                    # import pdb; pdb.set_trace()
                    startindex, endindx = text_cat_endindex[eachcat.lower()]
                    mask[tmpcnt, startindex:endindx] = 1
                cat_logit_per_image = cat_logit_per_image * mask

                gt_label = torch.argmax(cat_logit_per_image, dim=-1)

                for bb in range(len(englishname)):
                    pseudo_gt.append(text_prompts[gt_label[bb]])

        K=20
        imgnet_text = create_train_text_prompts_imagenet(pseudo_gt, englishname, attributes_all, categories_all, K=K)
        imgnet_text_tokens = clip.tokenize(imgnet_text).to(device)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        texts = create_train_text_prompts(cate_name, att_name, attributes_all, categories_all, K=K)
        # import pdb;
        # pdb.set_trace()

        text_tokens = clip.tokenize(texts).to(device)

        # print(images.min(), images.max())

        # with automatic mixed precision
        with autocast():
            with torch.no_grad():

                dr_text_emb = model_text(text_tokens)
                imgnet_text_emb = model_text(imgnet_text_tokens)

                dr_text_emb = dr_text_emb / dr_text_emb.norm(dim=1, keepdim=True)
                imgnet_text_emb = imgnet_text_emb / imgnet_text_emb.norm(dim=1, keepdim=True)


        with autocast():
            logit_scale = model.module.logit_scale.exp()

            prompted_images = prompter(clip_img_preprocessing(images))   # prompter(images)
            prompted_images_imgnet = prompter(clip_img_preprocessing(imgnetimg))
            prompt_token = add_prompter()
            # output, _ = model(prompted_images, text_tokens, prompt_token) ## only for single GPU

            if prompt_token is not None:
                bs = images.size(0)
                prompt_token = prompt_token.repeat(bs, 1, 1)

            drimg_fea = model_image(prompted_images, prompt_token=prompt_token)
            imgnet_fea = model_image(prompted_images_imgnet, prompt_token=prompt_token)

            drimg_fea = logit_scale * drimg_fea / drimg_fea.norm(dim=1, keepdim=True)
            imgnet_fea = logit_scale * imgnet_fea / imgnet_fea.norm(dim=1, keepdim=True)
            joint_fea = torch.cat([drimg_fea, imgnet_fea], dim=0)

            logit_dr = drimg_fea @ dr_text_emb.t()
            logit_img = imgnet_fea @ imgnet_text_emb.t()
            logit_joint = joint_fea @ cate_embedd.t()


            loss1 = criterion(logit_dr, target)
            loss2 = criterion(logit_img, target)
            loss3 = criterion(logit_joint, cat_target)

            # # for multiple GPU, inside it normalize, on doubly right dataset
            # output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)
            # loss1 = criterion(output, target)
            #
            # # on imagenet images,
            # output_imgnet, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images_imgnet,
            #                                  imgnet_text_tokens, prompt_token)
            # loss2 = criterion(output_imgnet, target)

            # # === classification only
            # image_fea = model_image(prompter(clip_img_preprocessing(imgnetimg)))
            # if args.use_loss_normalize:
            #     image_fea = image_fea / image_fea.norm(dim=1, keepdim=True)
            #     logit_scale = model.module.logit_scale.exp()
            #     image_fea = logit_scale * image_fea
            # import pdb; pdb.set_trace()

            loss = loss1 + loss2 + loss3
            if i% (args.print_freq*2) == 0:
                print('loss 1,2,3', loss1.item(), loss2.item(), loss3.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(logit_dr, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        imgnet_acc1 = accuracy(logit_img, target, topk=(1,))
        imgnet_top1.update(imgnet_acc1[0].item(), images.size(0))

        drclass_acc1 = accuracy(logit_joint[:bs], cat_target[:bs], topk=(1,))
        dr_classification_top1.update(drclass_acc1[0].item(), images.size(0))

        img_class_acc1 = accuracy(logit_joint[bs:], cat_target[bs:], topk=(1,))
        img_classification_top1.update(img_class_acc1[0].item(), images.size(0))

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

def dr_validate(model, prompter, add_prompter, criterion, args):
    import json
    with open('Our_ImageNet_attribute.json', 'r') as f:
        des = json.load(f)

    def construct_prompt(des, category):
        ans = []
        for each in des.keys():
            # ans.append(each)
            for com in des[each]:
                ans.append(f'This is a photo of a {category}, which is {com}')
        return ans

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])


    root_path = f'{args.Dr_rootpath}/test'
    Score_dict = {}
    Whole_score = 0
    BL_Whole_score = 0
    Whole_cnt = 0

    bl_Whole_score = 0
    for ii, category in enumerate(os.listdir(root_path)):
        # if category != 'syringe':
        #     continue
        num_of_attribute = len(os.listdir(os.path.join(root_path, category)))
        # we create the category combined with all 6k attributes, we then query the top k, where k equals to the number
        # of all the causal attributes relates to
        for attribute in os.listdir(os.path.join(root_path, category)):

            # try:

            # print('start', category, attribute)
            # val_dataset = torchvision.datasets.ImageFolder(
            #     os.path.join(root_path, category, attribute),
            #     transform=preprocess
            # )
            from dataloader import DataOneSet
            val_dataset = DataOneSet(os.path.join(root_path, category, attribute),
                transform=preprocess)

            train_sampler = None
            val_sampler = None

            val_loader = DataLoader(val_dataset,
                                    batch_size=min(len(val_dataset), 64), pin_memory=True,
                                    num_workers=1, shuffle=False, sampler=train_sampler, drop_last=True)

            Score_cnt = 0
            bl_Score_cnt = 0
            cnt = 0
            texts = construct_prompt(des, category)
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device)
                cnt += images.size(0)
                # target = target.to(device)
                text_tokens = clip.tokenize(texts).to(device)

                attribute = attribute.replace('__', '/')  # convert back to json's format.
                K = max(5, num_of_attribute * 2)

                with autocast():
                    # clean images, with prompt and without prompt
                    # compute output
                    with torch.no_grad():

                        prompt_token = add_prompter()
                        # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                        logits_per_image, _ = multiGPU_CLIP(None, None, model,
                                                         prompter(clip_img_preprocessing(images)), text_tokens,
                                                         prompt_token)

                        topk_value, topk_indices = torch.topk(logits_per_image, K, dim=1)
                        # we reterive 2 times of the possible attributes predicted by GPT. If any match, then it knows the reason.

                        for bb in range(topk_indices.size(0)):  # for each instance in batch
                            tops = []
                            for ll in range(topk_indices.size(1)):  # for each top k
                                tops.append(texts[topk_indices[bb, ll].item()])
                                if category.replace('_', ' ').replace('-', ' ').replace('/', ' ') in texts[topk_indices[bb, ll].item()] and attribute in texts[
                                    topk_indices[bb, ll].item()]:
                                    # if get both category and attribute correct
                                    # find specified category in top k reterival
                                    Score_cnt += 1

                        # =======================================
                        logits_per_image, _ = multiGPU_CLIP(None, None, model,
                                                            clip_img_preprocessing(images), text_tokens,
                                                            None)

                        topk_value, topk_indices = torch.topk(logits_per_image, K, dim=1)
                        # we reterive 2 times of the possible attributes predicted by GPT. If any match, then it knows the reason.

                        for bb in range(topk_indices.size(0)):  # for each intance in batch
                            tops = []
                            for ll in range(topk_indices.size(1)):  # for each top k
                                tops.append(texts[topk_indices[bb, ll].item()])
                                if category in texts[topk_indices[bb, ll].item()] and attribute in texts[
                                    topk_indices[bb, ll].item()]:
                                    # if get both category and attribute correct
                                    # find specified category in top k reterival
                                    bl_Score_cnt += 1


                            # TODO: for those fails on top k, print the predicted attribute and image.

            Score_dict[category + '-' + attribute] = Score_cnt / cnt


            Whole_score += Score_cnt
            Whole_cnt += cnt
            BL_Whole_score += bl_Score_cnt



        # print('\n\n\n\nresult', category + '-' + attribute, Score_cnt * 1.0 / cnt, 'baseline', bl_Score_cnt * 1.0 / cnt)
        if ii%50==0:
            print('ours', Whole_score/Whole_cnt, 'baseline', BL_Whole_score/Whole_cnt)

        if args.debug:
            break

    return Whole_score/Whole_cnt, BL_Whole_score/Whole_cnt



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

                # # generate adv example
                # if args.CW:
                #     delta_prompt = attack_CW(prompter, model, model_text, model_image, add_prompter, criterion,
                #                              images, target, text_tokens,
                #                              test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                # else:
                #     delta_prompt = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion,
                #                               images, target, text_tokens,
                #                               test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                #
                # # compute output
                # torch.cuda.empty_cache()
                # with torch.no_grad():
                #     prompt_token = add_prompter()
                #     # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
                #     output_prompt_adv, _ = multiGPU_CLIP(model_image, model_text, model,
                #                                          prompter(clip_img_preprocessing(images + delta_prompt)),
                #                                          text_tokens, prompt_token)
                #
                #     loss = criterion(output_prompt_adv, target)
                #
                # # bl attack
                # torch.cuda.empty_cache()
                #
                # if args.CW:
                #     delta_noprompt = attack_CW_noprompt(prompter, model, model_text, model_image, criterion,
                #                                         images, target, text_tokens, test_stepsize, args.test_numsteps,
                #                                         'l_inf', epsilon=args.test_eps)
                # else:
                #     delta_noprompt = attack_pgd_noprompt(prompter, model, model_text, model_image, criterion,
                #                                          images, target, text_tokens, test_stepsize, args.test_numsteps,
                #                                          'l_inf', epsilon=args.test_eps)
                # torch.cuda.empty_cache()
                # with torch.no_grad():
                #     # output_org_adv, _ = model(clip_img_preprocessing(images+delta_noprompt), text_tokens)
                #     output_org_adv, _ = multiGPU_CLIP(model_image, model_text, model,
                #                                       clip_img_preprocessing(images + delta_noprompt),
                #                                       text_tokens, None)

                # torch.cuda.empty_cache()
                # # measure accuracy and record loss
                # acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                # losses.update(loss.item(), images.size(0))
                # top1_adv_prompt.update(acc1[0].item(), images.size(0))
                #
                # acc1 = accuracy(output_org_adv, target, topk=(1,))
                # top1_adv_org.update(acc1[0].item(), images.size(0))

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
