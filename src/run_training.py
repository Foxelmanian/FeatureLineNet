from src import DeepLearningModels
import pandas as pd
import tensorflow as tf
from src import ClassicMachineLearningComparer

def get_model(do_training = True, machine_learning=False, database_path = '', name='', epochs=10):

    df_basic = pd.read_csv(database_path)
    length_of_data_base = len(df_basic)
    df_all = pd.read_csv(database_path)
    print('Create validation data')
    df_validation = df_basic[: int(0.2 * length_of_data_base)]
    df_basic = df_basic[int(0.2 * length_of_data_base):]
    print(f'Number of used train data {len(df_basic)}')

    # Lineare representation of the values
    if machine_learning:
        c1 = ClassicMachineLearningComparer.ClassicMachineLearningComparer()
        c1.compare_machine_learning_alrogithm(df_y=df_all[df_all.columns[0]],  df_x=df_all[df_all.columns[7:]], percentage=1.0 )

    # find number of categories
    y = df_basic['y']
    y_list = []
    for yy in y:
        if yy not in y_list:
            y_list.append(yy)
    number_of_categories = len(y_list)
    print(f'Number of categories {number_of_categories}: {y_list}')
    if do_training:
        # 2. Create the deep learning model or other model
        # Resotrt the data frame for the neuronal network
        train_data_basic, train_labels_basic = df_basic[df_basic.columns[7:]].values, df_basic[df_basic.columns[0]].values
        val_data, val_labels = df_validation[df_validation.columns[7:]].values, df_validation[df_validation.columns[0]].values
        deep_model_selecter = DeepLearningModels(train_data_basic, train_labels_basic, val_data, val_labels, number_of_categories, epochs=epochs)
        model = deep_model_selecter.train_model()
        model.save(f'my_model{number_of_categories}{name}.h5')
    else:
        model = tf.keras.models.load_model(f'my_model{number_of_categories}{name}.h5')
    return model, number_of_categories
