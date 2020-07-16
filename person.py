from openvino.inference_engine import IECore, IENetwork
import numpy as np
import sys
import cv2

CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib"

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        path = f"models/{model_name}/FP32"
        self.model_weights=f"{path}/{model_name}.bin"
        self.model_structure=f"{path}/{model_name}.xml"
        self.device=device
        self.threshold=threshold
        self.exec_net = None

        try:
            self.core = IECore()
            self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape

    def load_model(self):
        self.core.add_extension(extension_path=CPU_EXTENSION, device_name='CPU')
        self.exec_net = self.core.load_network(network=self.net, device_name=self.device, num_requests=1)

    def predict(self, image):
        image = self.preprocess_input(image)
        result = self.exec_net.infer({ self.input_name: image })
        coords = self.preprocess_outputs(result);
        image = self.draw_outputs(coords, image)

        return coords, image

    def draw_outputs(self, coords, image):
        for coord in coords[0][0]:
            if coord[2] >= self.threshold:
                x_min = int(coord[3])
                y_min = int(coord[4])
                x_max = int(coord[5])
                y_max = int(coord[6])

            image = cv2.rectangle(image, (x_min, x_max), (y_min, y_max), (255, 0, 0), 1)

        return image

    def preprocess_outputs(self, outputs):
        return outputs[self.output_name];

    def preprocess_input(self, image):
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, : , : , :]

        return image
