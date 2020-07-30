# smart-queue-system
Smart Queue System using Intel OpenVINO for inferencing. This uses queue regions to manage people count per queue.

## Setup

1. Install [OpenVINO Toolkit 2019 r3](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)
2. Setup environment using Python 3.6
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
```
3. Download the model for detecting a person: [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.4/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html).

## Running

There are 3 scenarios
- Manufacturing
- Retail
- Transportation

```
$ python3 main.py \
  --model person-detection-retail-0013 \
  --video videos/<scenario>.mp4 \
  --threshold 0.8 \
  --output_path results/<scenario> \
  --queue_param data/<scenario>.npy
```

Sample:

```
$ python3 main.py \
  --model person-detection-retail-0013 \
  --video videos/manufacturing.mp4 \
  --threshold 0.8 \
  --output_path results/manufacturing \
  --queue_param data/manufacturing.npy
```

## Output

### Manufacturing Scenario

Predict person location in frame to see if the production line is well-supported.

[![Watch the video](http://i3.ytimg.com/vi/OUEe3H4EjVk/hqdefault.jpg)](https://youtu.be/OUEe3H4EjVk)

Video: [Download](https://github.com/joseph-d-p/smart-queue-system/tree/master/results/manufacturing/output_video.mp4)

### Retail Scenario
Detect number of customers in queue and be able to redirect customer/s to less congested queues.

[![Watch the video](http://i3.ytimg.com/vi/pbHpxR9Yk_8/hqdefault.jpg)](https://www.youtube.com/watch?v=pbHpxR9Yk_8)

Video: [Download](https://github.com/joseph-d-p/smart-queue-system/tree/master/results/retail/output_video.mp4)


