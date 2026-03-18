# Image Caption Model

Image captioning project built on the Flickr8k dataset with VGG16 image features and sequence decoders in Jupyter notebooks.

This repository contains two notebooks:

- [Neural_Image_Caption_Generation.ipynb](/Applications/Python_AI/Neural_Image_Caption_Generation/Neural_Image_Caption_Generation.ipynb): the main working notebook, adapted to run locally in VS Code/Jupyter, with training, inference, beam search, and BLEU evaluation.
- [image_caption.ipynb](/Applications/Python_AI/Neural_Image_Caption_Generation/image_caption.ipynb): a more tutorial-style notebook with clearer step-by-step explanations.

## Project Goal

Generate natural language captions for input images from the Flickr8k dataset.

The main notebook currently includes:

- Flickr8k data loading and preprocessing
- VGG16 feature extraction
- Attention-based image captioning model using `features_conv`
- Training with checkpointing and early stopping
- Test inference on sample images
- BLEU-1 to BLEU-4 evaluation on the test set

## Current Best Direction

The strongest notebook version in this repo is the attention model in [Neural_Image_Caption_Generation.ipynb](/Applications/Python_AI/Neural_Image_Caption_Generation/Neural_Image_Caption_Generation.ipynb), which uses:

- VGG16 convolution features with shape `(49, 512)`
- an attention decoder
- sparse categorical loss
- streaming datasets to avoid high RAM usage

Recent attention-model results on the Flickr8k test set:

- `BLEU-1: 0.5119`
- `BLEU-2: 0.3175`
- `BLEU-3: 0.1899`
- `BLEU-4: 0.1065`

These results improved over the earlier FC-only LSTM baseline.

## Requirements

Recommended environment:

- macOS on Apple Silicon
- VS Code with Jupyter extension
- Python 3.10
- Conda environment or virtual environment

Main Python packages:

- `tensorflow`
- `tensorflow-metal` for Apple GPU acceleration
- `numpy`
- `pillow`
- `nltk`
- `matplotlib`
- `scipy`

## Running The Project

1. Clone the repository.
2. Open the project in VS Code.
3. Create/select a Python environment.
4. Open [Neural_Image_Caption_Generation.ipynb](/Applications/Python_AI/Neural_Image_Caption_Generation/Neural_Image_Caption_Generation.ipynb).
5. Select the correct Jupyter kernel.
6. Run the notebook cells in order.

The notebook is structured to:

- check the Python environment
- prepare local folders
- load or extract features
- build the captioning model
- train the model
- test predictions
- evaluate BLEU scores

## Notes

- Local datasets, trained weights, virtual environments, and IDE folders are ignored via `.gitignore`.
- The repo does not include the Flickr8k dataset or generated model files.
- If you train locally, your checkpoints will be stored under `workspace/models/` but are not committed.

## Future Improvements

- better attention mechanism
- pretrained word embeddings such as GloVe
- stronger CNN backbone than VGG16
- more robust decoding and qualitative evaluation

