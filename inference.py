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
    import random

    from dataset import RosbagDataset, load_rosbag_data

    bag_path = (
        "../dataset/rosbag2_2025_10_08-15_15_28/rosbag2_2025_10_08-15_15_28_0.db3"
    )
    scan_df, synchronized_joy_df = load_rosbag_data(bag_path)
    dataset = RosbagDataset(scan_df, synchronized_joy_df)

    num_samples = 10
    random_indices = random.sample(range(len(dataset)), num_samples)

    print("=" * 80)
    print(f"Random Sampling Test ({num_samples} samples)")
    print("=" * 80)

    onnx_path = "../model/model.onnx"
    model = Model(onnx_path)

    for i, idx in enumerate(random_indices, 1):
        input_ranges = dataset[idx][0].numpy()
        joy_speed_gt = dataset[idx][1].item()
        joy_angle_gt = dataset[idx][2].item()

        predicted_joy = model.inference(input_ranges)
        pred_speed, pred_angle = predicted_joy[0], predicted_joy[1]

        print(f"[Sample {i}] Index: {idx}")
        print(f"  Predicted  -> Speed: {pred_speed:7.4f}, Angle: {pred_angle:7.4f}")
        print(
            f"  Ground Truth -> Speed: {joy_speed_gt:7.4f}, Angle: {joy_angle_gt:7.4f}"
        )

    print("\n" + "=" * 80)
