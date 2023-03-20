# Pipeline for Doubly Right Object Recognition


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
