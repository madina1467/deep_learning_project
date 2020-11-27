# Hourglass Unet Bottleneck Network - TensorFlow 2
TensorFlow 2 implemetation of Hourglass Unet Bottleneck Network (Hourglass-104) with Multi-GPU training support. I've trained the model with MPII dataset. To train with another dataset, you will need to create TF Records and maybe change the preprocess.py file accordinly.
Hourglass-104 also contains implementation of Stacked Hourglass and Hourglass UNet Networks.

### MPII
Please download the dataset and follow the steps below in MPII directory to set up the dataset first.

## Set Up Dataset
move mpii images file to dataset file. So, it should look like:  tenserflow_pose/datasets/mpii/images. Now it consist only 5 images.
Then you need to create tfrecords od that dataset by running:
```
python3 tfrecords_mpii.py
```
If you want to use not all images of dataset dataset you can specify number of training and validation sets => change num_images_train and num_images_val inside tfrecords_mpii.py

## Start Training
Once all TF Records are generated, you could start training by:
```
python3 train.py
```

## Inference

Please see `demo_hourglass_pose.py` for examples of inference.
