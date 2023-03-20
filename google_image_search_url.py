import os
import shutil
import pandas as pd
import json
import glob
from PIL import Image

from google_images_search import GoogleImagesSearch
import uuid

from datetime import date

today = date.today()


# If you find our tools useful, please consider accept our paper, 

api_key = '' # you need your own key which setup in google account.
cx = ''
api_key_list=[api_key]

import socket

if 'cv' in socket.gethostname():


    # # SUN
    INIT_IMAGES_DOWNLOAD_DIR = 'DoubleRightDatasetOOD/SUN_dr'
    INIT_IMAGES_DOWNLOAD_URL_DIR = 'DoubleRightDatasetOOD/SUN_dr_url'

    with open('GPT3/SUN_text.json', 'r') as fp:
        attri_dict = json.load(fp)




os.makedirs(INIT_IMAGES_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(INIT_IMAGES_DOWNLOAD_URL_DIR, exist_ok=True)


img_att_url = {}

last_end=""
attribute_end=""
stop_at=""

api_cnt = 0

continue1=True
continue2=True

failure_cnt = 0

for category in attri_dict.keys():
    if stop_at == category:
        break
    #
    # # Start from where last time ends
    if category == last_end:
        continue1 = False
    if continue1:
        continue

    img_att_url[category] = {}
    api_key = api_key_list[api_cnt]
    att_list = attri_dict[category]

    for attributes in att_list:
        attributes = attributes.replace('/', ' or ').replace('-', ' ').replace('_', ' ')
        if attributes == attribute_end or attribute_end=='': # if not specified, then start from thefirst attribute
            continue2 = False
        if continue2:
            continue

        img_att_url[category][attributes] = []

        query_list = [f' A photo of {category} because there is {attributes}']
        for qi, query in enumerate(query_list):
            print('qi', qi, api_key)
            failure = True
            failure_cnt = 0

            while failure:

                _search_params = {
                'q': query,
                'num': 50,
                'safe': '',
                'fileType': '',
                'imgType': '',
                'imgSize': '',
                'imgDominantColor': '',
                'rights': ''
                }

                img_directory = os.path.join(INIT_IMAGES_DOWNLOAD_DIR, category, attributes, str(qi))
                os.makedirs(img_directory, exist_ok=True)

                url_directory = os.path.join(INIT_IMAGES_DOWNLOAD_URL_DIR, category, attributes, str(qi))
                os.makedirs(url_directory, exist_ok=True)

                # DEBUG
                print("obj_attr_img_directory", img_directory)
                print("query = ", _search_params['q'])

                # Init Google Image Search every iteration to avoid repetition error # trying
                gis = GoogleImagesSearch(api_key, cx)

                try :
                    gis.search(search_params=_search_params, custom_image_name="foo")
                except Exception as Error:
                    print ('error during query:', Error)
                    print('Query error with :', query)

                name_idx = 0
                for image in gis.results():

                    try:
                        image.download(img_directory)

                        # Also save URL
                        file1 = open(os.path.join(url_directory, f'img_url_{name_idx}.txt'), "w")
                        file1.writelines([image.path, '\n', image.url])
                        file1.close()
                        print('image url', image.url)
                        print('img path', image.path)

                        img_att_url[category][attributes].append(image.url)

                        # list_of_entity_url_tuples.append((entity_name, image.path, image.url))
                        # print(query, image.path, image.url)

                        # renamed_img_path = os.path.join(img_directory, f"image_{name_idx}.jpg")
                        name_idx += 1
                    except:
                        print('failed one')

                if name_idx > 0:
                    # success
                    failure = False
                else:
                    # failure
                    failure_cnt += 1
                    api_cnt += 1
                    api_key = api_key_list[api_cnt]
                    print("using api #", api_cnt, api_key)


                if failure_cnt > 2:
                    print("the last one is", category, attributes, query)
                    uuidname = str(uuid.uuid4())[:6]
                    with open(f'{today}_{uuidname}_unfinished_{category}_{attributes}.json', 'w') as f2:
                        json.dump(img_att_url, f2)
                    break
            if failure_cnt > 2:
                break
        if failure_cnt > 2:
            break

    if failure_cnt > 2:
        break

uuidname = str(uuid.uuid4())[:6]
with open(f'{today}_{uuidname}_finished.json', 'w') as f2:
    json.dump(img_att_url, f2)


