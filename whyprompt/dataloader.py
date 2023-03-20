
from torch.utils import data
import numpy as np
import os
import torchvision.datasets as datasets

from tqdm import tqdm
from PIL import Image
import random

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

class DataOneSet(data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.categories = os.listdir(path)
        self.categories.sort()

        self.all = []
        for each in self.categories:
            for img in os.listdir(os.path.join(path, each)):
                # if 'jpg' not in img:
                    # print(img)
                self.all.append(os.path.join(path, each, img))

        self.transform = transform

    def __getitem__(self, item):

        img_path = self.all[item]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert("RGB")

        # print('img load', img)
        sample = self._transform(img)
        # print('sample', sample.shape)
        img.close()

        return sample, 0

    def __len__(self):
        return len(self.all)

    def _transform(self, sample):
        return self.transform(sample)



class RandomLoader(data.Dataset):
    def __init__(self, path, composed_transforms=None):
        super().__init__()

        self.path = path
        self.categories = os.listdir(path)
        self.categories.sort()
        # self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}



        self.transform = composed_transforms

    def __getitem__(self, item):

        while True:
            select_category = random.sample(self.categories, 1)[0]
            cat_path = os.path.join(self.path, select_category)
            attribute_list = os.listdir(cat_path)
            selected_attribute = random.sample(attribute_list, 1)[0]

            # selected_attribute_post = selected_attribute.replace('/', )
            if ' and' == selected_attribute[-4:] or ' read' == selected_attribute[-5:]:
                tmp = os.listdir(os.path.join(cat_path, selected_attribute))[0]
                selected_attribute_name = selected_attribute + '/' + tmp

                att_path = os.path.join(cat_path, selected_attribute, tmp)
            else:
                att_path = os.path.join(cat_path, selected_attribute)
                selected_attribute_name = selected_attribute

            subf = os.listdir(att_path)
            if len(subf) < 1:
                continue
            s_subf = random.sample(subf, 1)[0]

            image_list = os.listdir(os.path.join(att_path, s_subf))
            if len(image_list) < 1:
                continue
            select_img = random.sample(image_list, 1)[0]

            img_path = os.path.join(att_path, s_subf, select_img)

            # print(selected_attribute)
            # try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample = self._transform(img)
            img.close()
            break


            # except:
            #     pass
        return sample, select_category.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower(), selected_attribute_name.replace('__', '/').replace('-', ' ').replace('/', ' or ').lower()   #replace('/', ' or ')

    def __len__(self):
        total = 0
        for root, dirs, files in os.walk(self.path):
            total += len(files)
        return total

    def _transform(self, sample):
        return self.transform(sample)


class RandomLoader_L2(data.Dataset):
    def __init__(self, path, composed_transforms=None):
        super().__init__()

        self.path = path
        self.categories = os.listdir(path)
        self.categories.sort()
        # self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}



        self.transform = composed_transforms

    def __getitem__(self, item):

        while True:
            select_category = random.sample(self.categories, 1)[0]
            cat_path = os.path.join(self.path, select_category)
            sup_attribute_list = os.listdir(cat_path)
            selected_sup_attribute = random.sample(sup_attribute_list, 1)[0]

            sup_att_path = os.path.join(cat_path, selected_sup_attribute)
            selected_sup_attribute_name = selected_sup_attribute

            attribute_list = os.listdir(sup_att_path)
            selected_attribute = random.sample(attribute_list, 1)[0]

            att_path = os.path.join(sup_att_path, selected_attribute)
            selected_attribute_name = selected_attribute

            # print("att_path: ", att_path)
            subf = os.listdir(att_path)
            if len(subf) < 1:
                continue
            s_subf = random.sample(subf, 1)[0]

            image_list = os.listdir(os.path.join(att_path, s_subf))
            if len(image_list) < 1:
                continue
            select_img = random.sample(image_list, 1)[0]

            img_path = os.path.join(att_path, s_subf, select_img)

            # # debug
            # print(img_path)
            # print(select_category.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower())
            # print(selected_sup_attribute_name.replace('__', '/').replace('-', ' ').replace('/', ' or ').lower())
            # print(selected_attribute_name.replace('__', '/').replace('-', ' ').replace('/', ' or ').lower())

            # print(selected_attribute)
            # try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample = self._transform(img)
            img.close()
            break


            # except:
            #     pass
        return sample, select_category.replace('_', ' ').replace('-', ' ').replace('/', ' or ').lower(), selected_sup_attribute_name.replace('__', '/').replace('-', ' ').replace('/', ' or ').lower(), selected_attribute_name.replace('__', '/').replace('-', ' ').replace('/', ' or ').lower()   #replace('/', ' or ')

    def __len__(self):
        total = 0
        for root, dirs, files in os.walk(self.path):
            total += len(files)
        return total

    def _transform(self, sample):
        return self.transform(sample)



class RandomLoader_combined(data.Dataset):
    def __init__(self, path, imagenet_path, composed_transforms=None):
        super().__init__()

        self.path = path
        self.imagenet_path = imagenet_path
        self.categories = os.listdir(path)
        self.categories.sort()
        # self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        from utils import load_imagenet_folder2name
        self.id2namemap = load_imagenet_folder2name('imagenet_classes_names.txt')
        self.imagenet_categories = os.listdir(imagenet_path)
        self.imagenet_categories.sort()

        self.transform = composed_transforms

    def __getitem__(self, item):

        while True:
            select_category = random.sample(self.categories, 1)[0]
            cat_path = os.path.join(self.path, select_category)
            attribute_list = os.listdir(cat_path)
            selected_attribute = random.sample(attribute_list, 1)[0]

            # selected_attribute_post = selected_attribute.replace('/', )
            if ' and' == selected_attribute[-4:] or ' read' == selected_attribute[-5:]:
                tmp = os.listdir(os.path.join(cat_path, selected_attribute))[0]
                selected_attribute_name = selected_attribute + '/' + tmp

                att_path = os.path.join(cat_path, selected_attribute, tmp)
            else:
                att_path = os.path.join(cat_path, selected_attribute)
                selected_attribute_name = selected_attribute

            subf = os.listdir(att_path)
            if len(subf) < 1:
                continue
            s_subf = random.sample(subf, 1)[0]

            image_list = os.listdir(os.path.join(att_path, s_subf))
            if len(image_list) < 1:
                continue
            select_img = random.sample(image_list, 1)[0]

            img_path = os.path.join(att_path, s_subf, select_img)

            # print(selected_attribute)
            # try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            sample = self._transform(img)
            img.close()
            break


        imgnet_select_category = random.sample(self.imagenet_categories, 1)[0]
        cat_path = os.path.join(self.imagenet_path, imgnet_select_category)

        englishname = self.id2namemap[imgnet_select_category]
        select_img = random.sample(os.listdir(cat_path), 1)[0]
        img_path = os.path.join(cat_path, select_img)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        imagenetimage = self._transform(img)


            # except:
            #     pass
        return sample, select_category.replace('_', ' ').replace('-', ' ').replace('/', ' ').lower(), selected_attribute_name.replace('__', '/'), imagenetimage, englishname.replace('_', ' ').replace('-', ' ').replace('/', ' ').lower()

    def __len__(self):
        return 1000000
        # return 256

    def _transform(self, sample):
        return self.transform(sample)

preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

if __name__ == "__main__":
    train_dataset = RandomLoader('/proj/vondrick4/chengzhi/DoubleRightShared/preprocessed/Image1kDR', composed_transforms=preprocess224_interpolate)
    train_dataset = RandomLoader_L2('/proj/vondrick4/chengzhi/DoubleRightDatasetOOD_split/CIFAR10_L2_v4_dr/train/', composed_transforms=preprocess224_interpolate)

    train_loader = DataLoader(train_dataset,
                              batch_size=16, pin_memory=True,
                              num_workers=2, shuffle=True, sampler=None)

    print(train_dataset[1])
    # for i, (images, cate_name, att_name) in enumerate(tqdm(train_loader)):
    #     print(images.shape, cate_name, att_name)
