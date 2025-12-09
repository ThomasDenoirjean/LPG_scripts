# Instructions

## List of concerned scripts

trainval_test_split.py
extract_inferences_result.py
get_test_metrics.py

## Scripts goal

ParticleTrieur 3.0.5 allows to split data into train and validation datasets, however the evaluation of model performance show ultimately be performed on a test set to ensure the model is evaluated on data it has nether see during training. With ParticleTrieur the validation data is monitored for its loss during training to update the learning rate of the model and to stop the training process. Thus, validation data plays a role during the model training and overfitting on validation data could not be excluded, which could harm model performance on unseen data.

Thus, we need to create a third dataset from the labelled data, the test set, which most of the time consist in 20% of the whole data available. We will evaluate the model performance thanks to the confusion matrix obtained on the test dataset while the validation confusion matrix help us to identify possible ways of improvement for the model training.

## How to use the scripts

First, copy all the files at the root of your ParticleTrieur project, open an anaconda prompt and navigate to the root of your project.

-1 Trainval test split

When you've labelled all your data, you can run the scripts to split your project in too, with the trainval datasets in one hand and the test set in the other.

To do so, run the following command in the anaconda prompt: python trainval_test_split.py <PROJECT_NAME>

The execution of the command will create 4 files:
- trainval.xml : the dataset for the training of the model
- test.xml : the dataset for the test of the model
- test_backup.xml : a backup of the test dataset. We will perform destructive operations on the test dataset during the evaluation, thus we may need this untouched backup
- test_labels.csv : a csv file with two rows : the picture GUID and its true label

The species with 11 specimens or less will be removed, to ensure that there is at least 10 specimens of each species in the trainval dataset.

Once this is done, open a new project and select the trainval file. You can now train your model.

Once you have a model you're satisfied with, open a new project and select the test file. Apply your model on all your pictures.

If you want to label more images for your model training, do it so on your main project, and once you finished just run the trainval_test_split.py script again.


-2 Extract the inference results

Once your model labelled all your test data, save your project and run this script : python extract_inferences_result.py

The execution of this script will create a new file, 'test_labels_w_pred.csv', which has three columns : the picture GUID, the true label and the predicted one.

This file can be processed with R to create your own metrics or graph.

If you would like to visualize some particular photos, for example to check the unusure/mispredicted pictures, on your project you can get to a specific picture thanks to its id by typing in the search bar : 'GUID==<PICTURE_GUID>'


-3 Get test metrics

Once you've extracted the inference results, run this script : python get_test_metrics.py

The base behavior of the script is to keep the unsure label ; if you don't want to include the data for which the predicted label is unsure, run the following command : python get_test_metrics.py --wo_unsure

This script will create the confusion matrix (unnormalised and normalised), display them and save them as 'cm.pdf' 'normalized_cm.pdf', respectively.

These are the metrics that give the best representation of your model performance: the data on which they were produced was never seen during its training and are brand new for it.