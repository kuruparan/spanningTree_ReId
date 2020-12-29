# veri-gms
Pytorch implementation of Vehicle Reidentification for Veri-776 using feature matching

Command to install the required libraries:

 	pip install -r requirements.txt  

You will also need to activate conda environment for pytorch.
Command to install the environment

	conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
(from: https://pytorch.org/)

GMS feature matcher: .pkl files for all the IDs of the cars in the dataset and its feature matches with each other.

Link: https://drive.google.com/drive/folders/15iQPcP8G45W4P84v2dKhzaO-lRbROLqo?usp=sharing

How to run:
1. Change the directory strings in main.py (line 112, 113, 168, 217, 242 and 262)

2. Choose the model architecture (line 154). If not listed in __init__.py in model folder, add it according to the format of others. 

3. Make sure you have the original dataset folder and the dataset with subfolders.
