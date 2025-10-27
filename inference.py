import numpy as np
import onnxruntime as ort


class Model:
    def __init__(self, path):
        self.ort_session = ort.InferenceSession(path)

    def inference(self, scan_ranges: np.ndarray) -> list:
        input_data = np.array(scan_ranges, dtype=np.float32).reshape(1, -1)

        ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
        ort_outs = self.ort_session.run(None, ort_inputs)

        joy_axes = ort_outs[0].flatten()

        return joy_axes.tolist()


if __name__ == "__main__":
    dummy_input = np.random.rand(1, 1081).astype(np.float32)
    model = Model("model.onnx")
    output = model.inference(dummy_input)
    print(output)
    