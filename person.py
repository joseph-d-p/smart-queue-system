from openvino.inference_engine import IECore

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model=core.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
    '''
    TODO: This method needs to be completed by you
    '''
        raise NotImplementedError

    def predict(self, image):
    '''
    TODO: This method needs to be completed by you
    '''
        raise NotImplementedError

    def draw_outputs(self, coords, image):
    '''
    TODO: This method needs to be completed by you
    '''
        raise NotImplementedError

    def preprocess_outputs(self, outputs):
    '''
    TODO: This method needs to be completed by you
    '''
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    TODO: This method needs to be completed by you
    '''
        raise NotImplementedError
