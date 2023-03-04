import os


data_dir = 'PATH_TO_STORE_DATA_ARTIFACTS'
dir_dreambooth = 'PATH_TO_train_dreambooth_mu'
path_labeled_img = os.path.join(data_dir, 'labeled_img_data.pkl')
path_labeled_labels = os.path.join(data_dir, 'labeled_img_labels.pkl')
path_clf_quality = os.path.join(data_dir, 'clfquality.pkl')