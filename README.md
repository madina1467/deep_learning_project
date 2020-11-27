# Hourglass Unet Bottleneck Network - TensorFlow 2
TensorFlow 2 implemetation of Hourglass Unet Bottleneck Network (Hourglass-104) with Multi-GPU training support. I've trained the model with MPII dataset. To train with another dataset, you will need to create TF Records and maybe change the preprocess.py file accordinly.
Hourglass-104 also contains implementation of Stacked Hourglass and Hourglass UNet Networks.


## Set Up Dataset
move mpii images file to dataset file. So, it should look like:  tenserflow_pose/datasets/mpii/images. Now it consist only 5 images.
Then you need to create tfrecords od that dataset by running:
```
python3 tfrecords_mpii.py
```
If you want to use not all images of dataset dataset you can specify number of training and validation sets => change num_images_train and num_images_val inside tfrecords_mpii.py
 
### MPII
Please follow [DATASET.md](../../Datasets/MPII/DATASET.md) in MPII directory to set up your dataset first. You could also use your own dataset, but this training scripts uses TF Records as data source, so you will need to generate TF Records in a similiar fashion.

## Start Training
Once all TF Records are generated, you could start training by:
```
python3 train.py
```

## Inference

Please see `demo_hourglass_pose.python` for examples of inference.
