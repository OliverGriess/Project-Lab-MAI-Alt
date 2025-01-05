# Install
Run:

````
pip install -r requirements.txt
````


Additionally you need to install pytorch.

Download the RenitaFace [ResNet50](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and move it to PRETRAIN/ALIGN.
Download [ICT Base](https://github.com/LightDXY/ICT_DeepFake/releases/download/v0.1.0/ICT_Base.pth) and add it to the checkpoints folder.

Create a directory called "pics" inside the DATASET directory.

Replace the preprocess.py in the ICT_DeepFake repo with preprocess_own.py.

Run preprocess_own.py.
