# Doubly Right Object Recognition: A Why Prompt for Visual Rationales (CVPR 2023)

<p align="center">
  <p align="center" margin-bottom="0px">
    <a href="http://www.cs.columbia.edu/~mcz/"><strong>Chengzhi Mao</strong></a>
    ·
    <a href=""><strong>Revant Teotia</strong></a>
    ·
    <a href=""><strong>Amrutha Sundar</strong></a>
    ·
    <a href=""><strong>Sachit Menon</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~junfeng/"><strong>Junfeng Yang</strong></a>
    ·
    <a href="https://xinw.ai/"><strong>Xin Wang</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~vondrick/"><strong>Carl Vondrick</strong></a></p>
    <p align="center" margin-top="0px"><a href="https://arxiv.org/abs/2212.07016">https://arxiv.org/abs/2212.07016</a></p>
</p>


# Doubly Right Object Recognition Benchmark

| Defence Method 	| Submitted By    	| DR Accuracy<br>(CIFAR10) | DR Accuracy<br>(CIFAR100) 	  | Submission Date 	|
|----------------	|-----------------	|----------------	|-----------------	|-----------------	|

|       Why Prompt        | (initial entry) 	|      |              |      Mar 1, 2023        |
|       CLIP        | (initial entry) 	|       |              |      Mar 1, 2023        |


## Data Collection
We provide the data collected through this procedure. For convenience, we provide a shortcut to those downloaded images [here](https://cv.cs.columbia.edu/mcz/DoubleRight.zip).
However, we do not own any of the images, and individual image will be deleted upon request.

You can also rerun the pipeline and recollect your own version of dataset.

1. Language Rationales

Run `GPT3/retrieve_language_rationale.py`, this requires the OpenAI API which need to pay.

We also include the retrieved rationales in the `GPT3` folder.

2. Language to Visual Rationales

`google_image_serach_url.py`, which ask Google what the language description look like visually, and
provides images downloaded as well as the url.

This requires the Google Image Search API which requires to pay.

## Train why prompt

See README.md in whyprompt.
