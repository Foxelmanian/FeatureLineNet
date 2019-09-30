# Polygon Feature Detection
# Author: Denk, Martin 27.05.2019, University of Applied Science Munich
# 

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

