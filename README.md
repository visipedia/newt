![Banner](assets/newt.jpeg)

# NeWT: Natural World Tasks

This repository contains resources for working with the NeWT dataset. 

## Benchmarking Representation Learning for Natural World Image Collections

Source code for reproducing the experiments in the [CVPR 2021 paper](https://arxiv.org/abs/2103.16483) can be found in the [benchmark](benchmark/) directory. 

## iNaturalist 2021 Dataset

The iNat2021 dataset is available [here](https://github.com/visipedia/inat_comp/tree/master/2021).

## NeWT 2021 Dataset

The NeWT 2021 dataset is a collection of 164 binary classification tasks. The images for all tasks have been combined into a single directory, and all images are stored as 3 channel jpegs. A single csv file contains the information for all tasks. See below for descriptions and download links. 

## Annotation Format

Image metadata and task data are stored in a csv file. Each row of the csv file corresponds to an image, and each image belongs to exactly 1 task. The csv has the  following columns:
  * `id`: the id of the image. Use this field to constuct the image filename by appending `.jpg`. 
  * `task_cluster`: the NeWT tasks are grouped into 5 clusters based on the classification problem (e.g. animal appearnce classification vs scene context).  
  * `task_subcluster`: further subdivides the NeWT tasks into additional clusters. Not all `task_clusters` have additional subclusters. 
  * `task`: the id of the task for this image. Use this field to select all images that belong to a given task. 
  * `label`: `0` or `1` label that corresponds to the binary label for this image. 
  * `text_label`: A human interpretable text label for the `0` and `1` binary label. 
  * `split`: either `train` or `test`. 
  * `height`: the height of the image, in pixels. 
  * `width`: the width of the image, in pixels. 

### Example Data

Example rows from `newt2021_labels.csv` with `task_subcluster` field:
| id                                   | task_cluster   | task_subcluster   | task                |   label | text_label   | split   |   height |   width |
|:-------------------------------------|:---------------|:------------------|:--------------------|--------:|:-------------|:--------|---------:|--------:|
| d80eb625-4982-4d34-ad9c-d957f565111e | appearance     | age               | ml_age_coopers_hawk |       0 | not_adult    | train   |      425 |     640 |
| 6c3eddb5-345f-444c-9957-1bee9c5aada2 | appearance     | age               | ml_age_coopers_hawk |       0 | not_adult    | train   |      430 |     640 |

The image files for these respective rows are accessible at:
  * `newt2021_images/d80eb625-4982-4d34-ad9c-d957f565111e.jpg`
  * `newt2021_images/6c3eddb5-345f-444c-9957-1bee9c5aada2.jpg`

<br>

Example rows from `newt2021_labels.csv` without `task_subcluster` field:

| id                                   | task_cluster   |   task_subcluster | task                       |   label | text_label   | split   |   height |   width |
|:-------------------------------------|:---------------|------------------:|:---------------------------|--------:|:-------------|:--------|---------:|--------:|
| 9a12feb3-bfbb-48cb-8227-21f5a8a4530c | context        |               nan | ml_bio_raptor_utility_pole |       0 | neg          | train   |      640 |     540 |
| 2cb20e06-9072-42fe-bd09-8557dfb591dc | context        |               nan | ml_bio_raptor_utility_pole |       0 | neg          | train   |      426 |     640 |

The image files for these respective rows are accessible at:
  * `newt2021_images/9a12feb3-bfbb-48cb-8227-21f5a8a4530c.jpg`
  * `newt2021_images/2cb20e06-9072-42fe-bd09-8557dfb591dc.jpg`

### Example Code
```python
import os

import imageio
import numpy as np
import pandas as pd

newt_labels_fp = "newt2021_labels.csv"
newt_image_dir = "newt2021_images/"

newt_df = pd.read_csv(newt_labels_fp)
newt_df['filepath'] = newt_df['id'].apply(
    lambda image_id: os.path.join(newt_image_dir, image_id + ".jpg")
)

task_results = []
for task_name, task_df in newt_df.groupby('task'):
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for _, image_row in task_df.iterrows():
        
        image = imageio.imread(image_row['filepath'])
        image_label = image_row['label']

        # Place holder for feature extraction
        image_feature = np.random.default_rng().uniform(-1,1,1000) 
        
        if image_row['split'] == 'train':
            X_train.append(image_feature)
            y_train.append(image_label)
        else:
            X_test.append(image_feature)
            y_test.append(image_label)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Placeholder for training and evaluating a classifier
    task_acc = np.random.default_rng().uniform(0,1,1)[0]
    
    task_results.append({
        'task' : task_name,
        'task_cluster' : task_df.iloc[0]['task_cluster'],
        'task_subcluster' : task_df.iloc[0]['task_subcluster'],
        'acc' : task_acc
    })

task_results_df = pd.DataFrame(task_results)
print(f"Mean Accuracy: {task_results_df['acc'].mean():.3f}")

for task_cluster, mean_acc in task_results_df.groupby('task_cluster')['acc'].mean().iteritems():
    print(f"{task_cluster:10} {mean_acc:0.3f}")
```


## Data Download

The dataset files are available through the AWS Open Data Program:

* [Images [4GB]](https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_images.tar.gz)
  * s3://ml-inat-competition-datasets/newt/newt2021_images.tar.gz
  * Running `md5sum newt2021_images.tar.gz` should produce `b04a56a5b1ffda87f16e6d4f81f9d38e`
  * All images are 3 channel jpegs. 

* [Labels & Metadata [1MB]](https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_labels.csv.tar.gz)
  * s3://ml-inat-competition-datasets/newt/newt2021_labels.csv.tar.gz
  * Running `md5sum newt2021_labels.csv.tar.gz` should produce `4cb26d0ee085904887b1ca14dcb893e7`


## Reference  
If you find our work useful in your research please consider citing our paper:  

```latex
@inproceedings{van2021benchmarking,
  title={Benchmarking Representation Learning for Natural World Image Collections},
  author={Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
```
