# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    # bbox: tlx, tly, drx, dry
    bbox = np.array(bbox)
    # bbox: tlx, tly, w, h
    bbox[2:] = bbox[2:] - bbox[:2]
    if patch_shape is not None:
        # TODO: 选择大者进行裁剪
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):
    def __init__(self,
                 checkpoint_filename,
                 input_name="images",
                 output_name="features"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(image_encoder, batch_size=32):
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255.,
                                          image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, image_dir, detections_in):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).

    """
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
    }

    detections_out = []

    frame_indices = detections_in[:, 0].astype(np.int) + 1
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]
        if frame_idx not in image_filenames:
            print("WARNING could not find image for frame %d" % frame_idx)
            continue
        bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
        # rows[1:5]: tlx, tly, drx, dry
        features = encoder(bgr_image, rows[:, 1:5].copy())
        detections_out += [
            np.r_[(row, feature)] for row, feature in zip(rows, features)
        ]

    return np.asarray(detections_out)


# def parse_args():
#     """Parse command line arguments.
#     """
#     parser = argparse.ArgumentParser(description="Re-ID feature extractor")
#     parser.add_argument(
#         "--model",
#         default="resources/networks/mars-small128.pb",
#         help="Path to freezed inference graph protobuf.")
#     parser.add_argument(
#         "--mot_dir", help="Path to MOTChallenge directory (train or test)",
#         required=True)
#     parser.add_argument(
#         "--detection_dir", help="Path to custom detections. Defaults to "
#         "standard MOT detections Directory structure should be the default "
#         "MOTChallenge structure: [sequence]/det/det.txt", default=None)
#     parser.add_argument(
#         "--output_dir", help="Output directory. Will be created if it does not"
#         " exist.", default="detections")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     encoder = create_box_encoder(args.model, batch_size=32)
#     generate_detections(encoder, args.mot_dir, args.output_dir,
#                         args.detection_dir)

# if __name__ == "__main__":
#     main()
