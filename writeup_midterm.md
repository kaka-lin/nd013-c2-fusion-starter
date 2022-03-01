# Sensor Fusion - 3D-Objects

In this section, I learned how to process lidar point-clouds data and detect objects from BEV images.

## Step 1. Converting Range Images to Point Clouds

### 1. Visualizing Range Images

Before we start to convert range images to point-clouds, we first visualize range image for understanding our dataset.

![](img/range_image.png)

### 2. Converting Range Images to Point Clouds

- Visualize the point cloud in Open3D
- 10 examples from point cloud with varying degrees of visibility


![](img/lidar_point_cloud_1.png)

![](img/lidar_point_cloud_2.png)

![](img/lidar_point_cloud_3.png)

![](img/lidar_point_cloud_4.png)

![](img/lidar_point_cloud_5.png)

![](img/lidar_point_cloud_6.png)

![](img/lidar_point_cloud_7.png)

![](img/lidar_point_cloud_8.png)

![](img/lidar_point_cloud_9.png)

![](img/lidar_point_cloud_10.png)


From these results, we can see the `rear-bumper` and `tail-lights` are the main stable features. In some cases, we can find `front-bumper`, `car headlights`, `license plate`, `windshields`, `sides of vehicles` also are stable features

## Step 2. Create Birds-Eye View (BEV) from Lidar PCL

### 1. Convert sensor coordinate to BEV-map coordinates

![](img/bev.png)


### 2. Compute intensity/hight/density layer of the BEV map

We can derive three pieces of information for each BEV cell, which are the `intensity of the points`, `their height`, and `their density`, hence the BEV map have three channels that make it a color image.

![](img/intensity_layer.png)

![](img/height_layer.png)

## Step 3. Model-based Object Detection in BEV Image

Feeding BEV-map into the model and extracting 3d bounding boxes from the model response.

![](img/labels_detected_objects.png)

## Step 4. Performance Evaluation for Object Detection

### 1. Compute intersection-over-union (IOU) between labels and detections

### 2. Compute false-negatives and false-positives

### 4. Compote percision and recall

The precision-recall curve is plotted showing similar results of `precision = 0.9506` and `recall = 0.944`.

![](img/performance_metric_01.png)

![](img/performance_metric_00.png)

In the next step, to make sure that the code produces plausible result, we set the

```python
configs_det.use_labels_as_objects=True
```

which results in `precision and recall values as 1`, This is shown in the following image:

![](img/performance_metric_11.png)

![](img/performance_metric_10.png)
