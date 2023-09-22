import torch
from clipfolder import clip
import os
import torchvision
from torch.utils.data import DataLoader
import tqdm
from torch.cuda.amp import GradScaler, autocast
import json
import matplotlib.pyplot as plt

from utils import getDictImageNetClasses
import numpy as np

# TODO: prompt engineering, better vocabulary,

which_layer=-1

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "RN50"
model_name = "RN101"
model_name = "ViT-B/16"
model_name = 'ViT-B/32'
model_name = "ViT-L/14"


model_name = "ViT-B/32"

model_names = ["RN50", "RN101", 'ViT-B/32', "ViT-B/16", "ViT-L/14"]

# dataset want to test
# name_test_list = ['cifar10'] # debug, use this for fast speed.
name_test_list=['SUN']
name_test_list=['cifar10', 'cifar100', 'food101_dr', 'caltech101', 'SUN', 'ImageNet']
# name_test_list = ['food101_dr', 'caltech101', 'cifar10', 'cifar100']

for model_name in model_names:

    model, preprocess = clip.load(model_name, device, jit=False)

    Img_MEAN = (0.48145466, 0.4578275, 0.40821073)
    Img_STD = (0.26862954, 0.26130258, 0.27577711)

    mu = torch.tensor(Img_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(Img_STD).view(3, 1, 1).cuda()

    def denormalize(X):
        return X * std + mu

    # model2, preprocess = clip.load('ViT-B/32', device, jit=False, extract_last_k_th_token=3)
    print('prprocess', preprocess)


    def construct_prompt(des):
        ans = []
        for each in des.keys():
            # ans.append(each)
            for com in des[each]:
                # ans.append(f'This is a photo of a {category}, which has {com}')
                ans.append(f'This is a photo of a {each} because there is {com}')
        return ans

    def construct_catonly_prompt(des):
        ans = []
        for each in des.keys():
            ans.append(f'This is a photo of a {each}')
        return ans


    def construct_prompt_full(des):
        all_att = []
        for each in des.keys():
            # ans.append(each)
            for com in des[each]:
                all_att.append(com)
        ans = []
        for each in des.keys():
            # ans.append(each)
            for com in all_att:
                # ans.append(f'This is a photo of a {category}, which has {com}')
                ans.append(f'This is a photo of a {each} because there is {com}')
        return ans

    import random
    def construct_prompt_partial(des):
        all_att = []
        for each in des.keys():
            # ans.append(each)
            for com in des[each]:
                all_att.append(com)

        ans = []
        for each in des.keys():
            # ans.append(each)
            for com in des[each]:
                # ans.append(f'This is a photo of a {category}, which has {com}')
                ans.append(f'This is a photo of a {each} because there is {com}')
            # partial random negative

            randsubset = random.sample(all_att, 10)
            for com in randsubset:
                ans.append(f'This is a photo of a {each} because there is {com}')

        return ans




    # TODO: ask clip to rank, if matches, score add one.


    # benchmarking clip via our dataset.
    root_path='/proj/vondrick4/chengzhi/DoubleRightDataset/init_downloaded_images_v1'
    root_path='/proj/vondrick4/chengzhi/DoubleRightShared/Image1kDR'
    root_path='/proj/vondrick4/chengzhi/DoubleRightShared/preprocessed/test'

    
    name_list=['ImageNet']
    root_path_list=['/proj/vondrick4/chengzhi/DoubleRightShared/deduplicated/test_convert/test']
    json_list = ['Our_ImageNet_attribute.json']


    for name, root_path, json_path in zip(name_list, root_path_list, json_list):
        if name not in name_test_list:
            continue


        with open(json_path, 'r') as f:
            des = json.load(f)

        RR_Score_dict={}
        RW_Score_dict={}
        WR_Score_dict={}
        WW_Score_dict={}
        Acc_dict = {}

        RR_Whole_score=0
        RW_Whole_score=0
        WR_Whole_score=0
        WW_Whole_score=0
        cat_correct_whole=0
        Whole_cnt=0

        Save_images = False
        save_path = '/proj/vondrick4/chengzhi/DoubleRightShared/Viz_DR'



        # texts = construct_prompt(des) # this is easier version
        texts = construct_prompt_full(des)
        # texts = construct_prompt_partial(des)
        # print(texts)
        block_size = 5000
        print(name, len(texts))
        if len(texts)<block_size:
            with autocast():
                with torch.no_grad():
                    cat_text_tokens = clip.tokenize(texts).to(device)
                    catatt_text_emb = model.encode_text(cat_text_tokens)
                    catatt_text_emb = catatt_text_emb / catatt_text_emb.norm(dim=-1, keepdim=True)
        else:
            block_num = len(texts) // block_size + 1
            print('block num', block_num)
            catatt_text_emb = []
            with autocast():
                with torch.no_grad():
                    for ind in range(block_num):
                        cat_text_tokens_tmp = clip.tokenize(texts[ind*block_size:ind*block_size+block_size]).to(device)
                        catatt_text_emb_tmp = model.encode_text(cat_text_tokens_tmp)
                        catatt_text_emb_tmp = catatt_text_emb_tmp / catatt_text_emb_tmp.norm(dim=-1, keepdim=True)
                        catatt_text_emb.append(catatt_text_emb_tmp)

            catatt_text_emb = torch.cat(catatt_text_emb, dim=0)


        # category only
        cate_prompt = construct_catonly_prompt(des)
        with autocast():
            with torch.no_grad():
                cat_text_tokens = clip.tokenize(cate_prompt).to(device)
                cat_text_emb = model.encode_text(cat_text_tokens)
                cat_text_emb = cat_text_emb / cat_text_emb.norm(dim=-1, keepdim=True)


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
                        # print('mg', att_dir, os.listdir(att_dir))

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

                    # print('att_dir', att_dir)
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
                    category = category_ori.replace('__', '/') # for SUN dataset
                    category = category.replace('_', ' ')


                    if Save_images:
                        os.makedirs(os.path.join(save_path, category, attribute.replace('/', '__')), exist_ok=True)
                        imgcnt = 0

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
                                image_fea = model.encode_image(images, ind_prompt=None, local_ind_prompt=None)
                                image_fea /= image_fea.norm(dim=-1, keepdim=True)

                                logits_per_image = image_fea @ catatt_text_emb.t()
                                cat_logits_per_img = image_fea @ cat_text_emb.t()

                                _, topk_cat_indices = torch.topk(logits_per_image, 5, dim=1)

                                _, topk_catonly_indices = torch.topk(cat_logits_per_img, 1, dim=1)


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
                                        correct_category_bool = True
                                    # right reason help get the right prediction
                                    if prediction == category_ori and together_reason_right:
                                        reason_right = True
                                        RR_Score_cnt += 1
                                    # prediction right, but none reason is right
                                    elif prediction == category_ori:
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

                                    # import pdb;
                                    # pdb.set_trace()


                        if Save_images:
                            for bb in range(topk_cat_indices.size(0)):  # for each intance in batch
                                imgcnt += 1
                                text_str = ''
                                for ll in range(topk_cat_indices.size(1)):  # for each top k
                                    text_str = text_str + '\n' + texts[topk_cat_indices[bb, ll].item()]

                                f, axarr = plt.subplots(2)
                                # import pdb; pdb.set_trace()

                                axarr[0].imshow(torch.permute(denormalize(images[bb]), (1, 2, 0)).cpu().numpy())
                                axarr[1].text(0, 0, text_str)

                                plt.savefig(os.path.join(save_path, category, attribute.replace('/', '__'), f'img{imgcnt}.jpg'))



                    RR_Score_dict[category + '-' + attribute] = RR_Score_cnt / cnt
                    RW_Score_dict[category + '-' + attribute] = RW_Score_cnt / cnt
                    WR_Score_dict[category + '-' + attribute] = WR_Score_cnt / cnt
                    WW_Score_dict[category + '-' + attribute] = WW_Score_cnt / cnt
                    Acc_dict[category + '-' + attribute] = cat_correct / cnt
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

                

        print(model_name, name, 'total score', RR_Whole_score / Whole_cnt, RW_Whole_score/Whole_cnt, WR_Whole_score/Whole_cnt,
              WW_Whole_score/Whole_cnt, cat_correct_whole/Whole_cnt)

       

