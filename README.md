Image-Captioner-Attention

This is a [PyTorch](https://pytorch.org) implementation of a topic driven image captioning system.
The model takes in an input image and a topic and tries to generate a caption that is topic focused and image accurate.

## Prerequisites
* Python 2.7
* PyTorch 0.4
* tqdm 4.2.3
* CUDA 8.0 or higher

## Preparation
1. Clone the repository
`git clone https://github.com/Andrew03/Controllable-Image-Captioning && cd Controllable-Image-Captioning`
2. Create the `data`, the `data/images` and the `data/datasets` directory
`mkdir data && mkdir data/images && mkdir data/datasets`
3. Create a softlink to the raw data 
`ln -s $RAW_DATA data/raw`
4. Download the images (This while take a while)
`python tools/download_images.py`
5. Process the data and build a vocabulary
`python tools/format_data.py`
