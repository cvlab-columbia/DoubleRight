import os
import openai
import json

import itertools


def generate_prompt(category_name: str):
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""


def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]

def smart_obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}

    # prompts = [generate_prompt_shots(category, shots, ['iphone', 'aardvark']) for category in class_list]
    prompts = [generate_prompt(category) for category in class_list]

    responses = [openai.Completion.create(model="text-davinci-002",
                                          prompt=prompt_partition,
                                          temperature=0.,
                                          max_tokens=100,
                                          ) for prompt_partition in partition(prompts, 10)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)



import os
import openai
from utils import getListImageNetClasses
import json
import uuid


# Load your API key from an environment variable or secret management service
# You need to setup openai account for this key
openai.api_key = os.getenv("OPENAI_API_KEY")


# class_names = getListImageNetClasses()

from dataset_others.caltech import *

class_names = caltech
class_names = [each.replace('A photo of a ', '').replace('.', '') for each in class_names]

class_names = cifar10

class_names = cifar100

class_names = food101
class_names = [each.replace('A photo of ', '').replace(', a type of food.', '') for each in class_names]


class_names = SUN

class_names = caltech101
class_names = [each.replace('A photo of a ', '').replace('.', '') for each in class_names]

import json
uuidname = str(uuid.uuid4())[:6]

smart_obtain_descriptors_and_save(f'caltech101_text.json', class_names)



