from layers import Conv3x3, MaxPool, Dense
from PIL import Image
import numpy as np
import time
import os
from utils import apply_patches, apply_shaped_mask, extract_faces
from matplotlib import pyplot as plt


class Generator:
    def __init__(self):
        self.net = [
            Conv3x3(4),
            MaxPool(),
            Conv3x3(2),
            Dense(29*29*8, 70*80),
            Dense(70*80, 50*60, activation="tanh")
        ]

    def forward(self, face):
        """
        :param face: 1d np array of shape (128, 128)
        :param shape: reversed tuple of patch shape
        :return: generated patch
        """
        out = face
        for layer in self.net:
            out = layer.forward(out)
        return out.reshape(60, 50)

    def backprop(self, loss_grad):
        loss_grad = self.net[4].backprop(loss_grad, .005)
        loss_grad = self.net[3].backprop(loss_grad, .005)
        loss_grad = self.net[2].backprop(loss_grad, .005)
        loss_grad = self.net[1].backprop(loss_grad)
        loss_grad = self.net[0].backprop(loss_grad, .005)


class Predictor:
    def __init__(self):
        self.net = [
            Dense(50*60, 60*60),
            Conv3x3(4),
            MaxPool(),
            Dense(29*29*4, 1, activation="sigmoid")
        ]

    def forward(self, face):
        """
        :param face: 1d np array of shape (128, 128)
        :param shape: reversed tuple of patch shape
        :return: generated patch
        """
        out = face
        out = self.net[0].forward(out)
        out = np.reshape(out, (60, 60))
        out = self.net[1].forward(out)
        out = self.net[2].forward(out)
        out = self.net[3].forward(out)
        return out

    def backprop(self, loss_grad):
        loss_grad = self.net[3].backprop(loss_grad, .005)
        loss_grad = self.net[2].backprop(loss_grad)
        loss_grad = self.net[1].backprop(loss_grad, .005)
        loss_grad = self.net[0].backprop(loss_grad, .005)
        return loss_grad


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def discriminator(patches, src_img, face_attr):
    """
    :param patches: PIL img - patches
    :param src_img: PIL img with some faces in it
    :param face_attr: attributes of the current processed face
    :return: verdict
    """
    overlap_threshold = 0.8
    # lpatch, rpatch = [Image.fromarray(np.resize(i, (int(GENERATE_SQUARE/2), GENERATE_SQUARE))) for i in np.split(patches, 2)]
    box, probs, points = face_attr

    patched_img = src_img.copy()
    face_bound = box.astype(int).tolist()
    face = src_img.crop(face_bound)

    face_area = (box[2]-box[0]) * (box[3]-box[1])
    patched_face = apply_patches(face, patches, patches, points, face_bound)
    patched_img.paste(apply_shaped_mask(face, patched_face, points), face_bound)

    box_p, probs_p, points_p = extract_faces(patched_img, keep_probs=True)
    if box_p is None:
        return 1  # Face not found (and it was the only face)
    for i in range(len(box_p)):
        overlap_width = min(box[2], box_p[i][2]) - max(box[0], box_p[i][0])
        overlap_height = min(box[3], box_p[i][3]) - max(box[1], box_p[i][1])
        if overlap_width > 0 and overlap_height > 0:
            overlap = overlap_width * overlap_height
            if overlap/face_area > overlap_threshold:
                return 1 - (probs_p[i] - 0.89) * 9.090909090909092  # Where 0.89 - min threshold for face recognition net
    return 1  # Face not found


generator = Generator()
predictor = Predictor()


def forward_step(photo, face_attributes):
    face_bound = face_attributes[0].astype(int).tolist()
    face = photo.crop(face_bound).resize((64, 64), Image.LANCZOS).convert("L")
    face = np.asarray(face).astype(np.float32)
    face = face / 255  # 127.5 - 1.

    return generator.forward(face)


def train_step(photo, face_attributes):
    """
    :param photo: source photo
    :param face_attributes: attributes of current processing face
    :return: binary cross entropy loss
    """
    src_img = photo
    generated_patches = forward_step(src_img, face_attributes)
    prediction = predictor.forward(generated_patches)
    # plt.imshow(generated_patches)
    # plt.show()

    patches = np.uint8((generated_patches+1) * 127.5)
    patches = Image.fromarray(patches).convert('RGBA')
    decision = discriminator(patches, src_img, face_attributes)

    # Apply binary cross entropy derivative
    loss_gradient = -(decision/prediction - (1-decision)/(1-prediction))
    loss_gradient = predictor.backprop(loss_gradient)
    generator.backprop(loss_gradient)

    return -decision * np.log(prediction) - (1-decision) * np.log(1 - prediction)


def train(datapath, epochs):
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        gen_loss_list = []

        for filename in os.listdir(datapath):
            path = os.path.join(datapath, filename)
            photo = Image.open(path)
            boxes, probs, points = extract_faces(photo, keep_probs=True)  # TODO: Shall I catch no-face images?
            for i, face_attributes in enumerate(zip(boxes, probs, points)):
                t = train_step(photo, face_attributes)
                gen_loss_list.append(t[0])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)

        epoch_elapsed = time.time() - epoch_start
        print(f'Epoch {epoch + 1:2}, gen loss={g_loss:.12f}, {hms_string(epoch_elapsed)}')
        if not (epoch + 1) % 5:
            process_photo("photos/photo_3.jpg", f"epoches/epoch_{epoch + 1}")

    elapsed = time.time() - start
    print(f'Training time: {hms_string(elapsed)}')


def process_photo(photo, name=None):
    if isinstance(photo, str):
        img = Image.open(photo)
    elif isinstance(photo, Image.Image):
        img = photo.copy()
    boxes, probs, points = extract_faces(img, keep_probs=True)
    patched_img = img.copy()
    for i, (box, point) in enumerate(zip(boxes, points)):
        face_bound = box.astype(int).tolist()
        face = img.crop(face_bound)
        patches = forward_step(img, (box, point))
        patches = np.uint8((patches + 1) * 127.5)
        patches = Image.fromarray(patches).convert('RGBA')

        patched_face = apply_patches(face, patches, patches, point, face_bound)
        patched_img.paste(apply_shaped_mask(face, patched_face, point), face_bound)
    if name:
        patched_img.save(f"{name}.png")
    return patched_img


if __name__ == "__main__":
    # conv = Conv3x3(8)
    # mpool = MaxPool()
    # dense = Dense(97*71*8, 33*14, activation="relu")
    # out = conv.forward(face)
    # out = mpool.forward(out)
    # out = dense.forward(out)
    # out = out.reshape(14, 33)
    # gradient = np.random.rand(33, 14)
    # gradient = -1 / gradient
    # gradient = dense.backprop(gradient, .005)
    # gradient = mpool.backprop(gradient)
    # gradient = conv.backprop(gradient, .005)
    train('./photos', 75)
