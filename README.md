# FeatureLineNet





<p align="center">
  <img src="https://github.com/DMST1990/FeatureLineNet/blob/master/documentation/UseCaseExample.png" width="75%">
</p>

## Articel

The articel [article](https://doi.org/10.35199/dfx2019.21) is under open access.
With the title

-Feature line detection of noisy triangulated CSGbased objects using deep learning


## Citation
Citate this work with

```
@inproceedings{denk_feature_2019,
	address = {Jesteburg, Germany},
	series = {{DfX}},
	title = {Feature line detection of noisy triangulated {CSGbased} objects using deep learning},
	volume = {30},
	url = {https://www.designsociety.org/publication/42428/Feature+line+detection+of+noisy+triangulated+CSGbased+objects+using+deep+learning},
	doi = {10.35199/dfx2019.21},
	abstract = {Feature lines such as sharp edges are the main characteristic lines of a surface. These lines are suitable as a basis for surface reconstruction and reverse engineering [1]. A supervised deep learning approach based on graph convolutional networks on estimating local feature lines will be introduced in the following. We test this deep learning architecture on two provided data sets of which one covers sharp feature lines and the other arbitrary feature lines based on unnoisy meshed constructive solid geometry (CSG). Furthermore, we use a data balancing strategy by classifying different feature line types. We then compare the selected architecture with classical machine learning models. Finally, we show the detection of these lines on noisy and deformed meshes.},
	language = {en},
	urldate = {2019-10-02},
	booktitle = {Proceedings of the 30th {Symposium} {Design} for {X}},
	publisher = {The Design Society},
	author = {Denk, Martin and Paetzold, Kristin and Rother, Klemens},
	month = sep,
	year = {2019},
	pages = {239--250},
}
```

## Introduction

FeatureLineNet is an open source deep learning appraoch for detecting feature lines on triangulated meshed data.
This software runs on Windows, OS X and Linux.
FeatureLineNet has been developed by Martin Denk and Prof. Dr. Rother Klemens at the [University of Applied Science Munich](https://www.hm.edu/) at the institute for material and building research.


## Current Version
- 3D triangulated Meshes
- Detection of edges in noisy objects
- Usuage of a simple graph convolutional neuronal network (GCN)
- Keras implementation

## Requierements
- Keras
- Pandas
- Numpy


## Call the function

```bash,run from commandline

python main.py

```

## Change the data set and hyperparameters
- Two data bases 'edges', 'primitive' are available
- You can switch between them by choosing the propper Training data base 'feature_db'
- You can run several machine leraning models or choos our GCN network
- 

```python,Chan

import os
import pandas as pd
from src import FeatureVisualizer
from src import run_training

#1.------
# First data set only primitives and simple CSG objects
#feature_db = os.path.join('Data', 'PrimitiveData.feat')
#name = 'primitive'
# Second data set primitives simple CSG objects and bevel, rounded ... corners
feature_db = os.path.join('Data', 'EdgeData.feat')
name = 'edge'

#2.------
# Training of the models
do_training = False
machine_learning = False
model_ft, number_of_categories_ft = run_training.get_model(do_training=do_training,
                                                           machine_learning=machine_learning,
                                                           database_path=feature_db,
                                                           name=name,
                                                           epochs=1000)
if number_of_categories_ft != 2:
    raise ValueError(f'Binary classifciation requies two categories not {number_of_categories_ft}')

#3.------
# View unlabeled data via mayavi
for stl_file in os.listdir(os.path.join('Data', 'UnlabeledData')):
    if stl_file.endswith(".stl"):
        file_name = stl_file[0:-4]
        visualizer = FeatureVisualizer(model_ft, number_of_categories_ft)
        feature_data_frame = visualizer.plot(os.path.join('Data', 'UnlabeledData',  file_name + '.stl'),
                                          df=pd.read_csv(os.path.join('Data', 'UnlabeledData', file_name + '.ftu')))


```





