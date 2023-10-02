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
  * `text_label`: A human interpretable text label for the `0` and `1` binary label. NOTE: these are *not* unique across tasks. Use the `task` field as the unique identifier for a task. 
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

The following example code demonstates how to iterate over the 164 binary classification tasks. There are placeholder commands for extracting features from the images and training binary classifiers. 

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

The following code block demonstrates how to iterate over the tasks within each task cluster and subcluster. Partial output is provided after the code block.

```python
import os

import pandas as pd

####
# Load in the NeWT dataset
newt_labels_fp = "newt2021_labels.csv"
newt_image_dir = "newt2021_images/"

newt_df = pd.read_csv(newt_labels_fp)
newt_df['filepath'] = newt_df['id'].apply(
    lambda image_id: os.path.join(newt_image_dir, image_id + ".jpg")
)

####
# Print some basic info for the task clusters
newt_task_clusters = newt_df['task_cluster'].unique()
print("NeWT Task Clusters")
for i, task_cluster_name in enumerate(newt_task_clusters):
    task_cluster_df = newt_df[newt_df['task_cluster'] == task_cluster_name]
    task_subclusters = task_cluster_df['task_subcluster'].dropna().unique()
    num_subclusters = task_subclusters.shape[0]
    num_tasks = task_cluster_df['task'].unique().shape[0]
    print(f"{i+1:1d}. {task_cluster_name:10s} ({num_subclusters:2d} subclusters) : {num_tasks:3d} tasks")
    if num_subclusters > 0:
        for task_subcluster_name in task_subclusters:
            task_subcluster_df = task_cluster_df[task_cluster_df['task_subcluster'] == task_subcluster_name]
            num_tasks = task_subcluster_df['task'].unique().shape[0]
            print(f"\t{task_subcluster_name:10s}: {num_tasks:3d} tasks")
    print()
print()
        
####
# Choose a specific task cluster and print basic info about the tasks
task_cluster_name = 'appearance'
assert task_cluster_name in newt_task_clusters, f"{task_cluster_name} is not a valid NeWT task cluster"
task_cluster_df = newt_df[newt_df['task_cluster'] == task_cluster_name]

num_subclusters = task_cluster_df['task_subcluster'].dropna().unique().shape[0]
num_tasks = task_cluster_df['task'].unique().shape[0]

if num_subclusters > 0:
    print(f"The task `{task_cluster_name}` has {num_tasks:3d} total tasks, grouped into {num_subclusters} subclusters")
else:
    print(f"The task `{task_cluster_name}` has {num_tasks:3d} total tasks")
    

print(f"Tasks included in the `{task_cluster_name}` cluster:")
for i, (task_name, task_df) in enumerate(task_cluster_df.groupby('task')):
    
    print(f"{i+1:3d}. {task_name:55s}")
    
    # Get positive and negative examples for this task
    task_positive_df = task_df[task_df['label'] == 1]
    task_negative_df = task_df[task_df['label'] == 0]
    
    # Get the text label for the positive and negative classes
    # NOTE: these are not unique across the tasks. 
    positive_text_label = task_positive_df.iloc[0]['text_label']
    negative_text_label = task_negative_df.iloc[0]['text_label']
    print(f"\tpositive class text label: {positive_text_label}")
    print(f"\tnegative class text label: {negative_text_label}")
    print(f"\tpositive images: {task_positive_df.shape[0]:3d}, negative images: {task_negative_df.shape[0]:3d}")
    print()
    
    # Get `train` and `test` examples for this task
    for split in ['train', 'test']:
        task_split_df = task_df[task_df['split'] == split]
        
        task_split_pos_df = task_split_df[task_split_df['label'] == 1]
        task_split_neg_df = task_split_df[task_split_df['label'] == 0]
        
        print(f"\tpositive {split:5s} images: {task_split_pos_df.shape[0]:3d}")
        print(f"\tnegative {split:5s} images: {task_split_neg_df.shape[0]:3d}")
        print()
```
Example output from the above code block:
```
NeWT Task Clusters
1. context    ( 0 subclusters) :   8 tasks

2. appearance ( 4 subclusters) : 132 tasks
   	attribute :   7 tasks
   	age       :  14 tasks
   	species   : 102 tasks
   	health    :   9 tasks

3. gestalt    ( 0 subclusters) :   6 tasks

4. behavior   ( 0 subclusters) :  16 tasks

5. counting   ( 0 subclusters) :   2 tasks


The task `appearance` has 132 total tasks, grouped into 4 subclusters
Tasks included in the `appearance` cluster:
  1. fgvcx_icassava_healthy_vs_sick                         
   	positive class text label: sick
   	negative class text label: healthy
   	positive images: 200, negative images: 200
   
   	positive train images: 100
   	negative train images: 100
   
   	positive test  images: 100
   	negative test  images: 100

  2. fgvcx_plant_pathology_healthy_vs_sick                  
   	positive class text label: sick
   	negative class text label: healthy
   	positive images: 200, negative images: 200
   
   	positive train images: 100
   	negative train images: 100
   
   	positive test  images: 100
   	negative test  images: 100

  3. inat_non_species_black_eastern_gray_squirrel           
   	positive class text label: black_squirrel
   	negative class text label: regular_squirrel
   	positive images:  77, negative images: 100
   
   	positive train images:  50
   	negative train images:  50
   
   	positive test  images:  27
   	negative test  images:  50

  4. inat_non_species_dead_common_garter_snake              
   	positive class text label: common_garter_snake
   	negative class text label: gopher_snake
   	positive images: 100, negative images: 100
   
   	positive train images:  50
   	negative train images:  50
   
   	positive test  images:  50
   	negative test  images:  50
...
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
