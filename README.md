# smart-queue-system
Smart Queue System using Intel OpenVINO for inferencing.

### Model

Using Intel's [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.4/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model.

### Running
```
$ python3 main.py \
  --model person-detection-retail-0013 \
  --video videos/manufacturing.mp4 \
  --threshold 0.8 \
  --output_path results/manufacturing
```

### Output

Manufacturing Video: [Download](https://github.com/joseph-d-p/smart-queue-system/tree/master/results/manufacturing)

[![Watch the video](http://i3.ytimg.com/vi/pvxg00jGmq8/hqdefault.jpg)](https://youtu.be/pvxg00jGmq8)

