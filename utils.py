from PIL import Image, ImageDraw, ImageFilter, ImageFont
from facenet_pytorch import MTCNN
from math import atan, pi, fabs
# import argparse


def extract_faces(img, keep_probs=False):
    """
    Find faces on img with facenet_pytorch MTCNN model
    :param img: PIL Image
    :param keep_probs: return or not face probabilities
    :return: attributes of all detected faces
    """
    mtcnn = MTCNN(keep_all=True)
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    if keep_probs:
        return boxes, probs, points
    else:
        return boxes, points


def add_caption(img, text):
    """
    :param img: PIL image
    :param text: plaintext
    :return: image with caption (black stripe on bottom with text)
    """
    result = img.copy()
    font = ImageFont.truetype("times", 64)
    strip_width, strip_height = img.width, 60
    background_strip = Image.new('RGB', (strip_width, strip_height))
    draw = ImageDraw.Draw(background_strip)
    while draw.textsize(text, font)[0] > img.width*0.97:    # Guessing correct font size
        font = ImageFont.truetype("times", font.size-2)
    text_width, text_height = draw.textsize(text, font)
    position = ((strip_width - text_width) / 2, (strip_height - text_height))
    draw.text(position, text, (255, 255, 255), font=font)
    result.paste(background_strip, (0, int(img.height-strip_height*1.5)))
    return result


def draw_landmarks(box, point, img, debug_info=False, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    draw.rectangle(box.tolist(), width=3, outline=color)
    for p in point:
        draw.ellipse((p - 2).tolist() + (p + 2).tolist(), width=3, outline=color)
        if debug_info:
            draw.text((box[0], box[3]+3), f"{int(box[2]-box[0])}x{int(box[3]-box[1])}", (255, 0, 0))
            dist_between_eyes = (point[1][1] - point[0][1])
            angle = -atan(dist_between_eyes / (point[1][0] - point[0][0])) * 180 / pi
            draw.text((box[0], box[3]+12), f"{round(angle, 2)}°", (255, 0, 0))
            draw.text((box[0], box[3]+21), f"hCorr {int(dist_between_eyes * 0.3)}", (255, 0, 0))


def apply_shaped_mask(face_source, face_perturbed, point):
    mask = Image.new("L", face_source.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0) + face_source.size, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(11))
    angle = -atan((point[1][1] - point[0][1]) / (point[1][0] - point[0][0])) * 180 / pi  # rotate mask by level of eyes
    result = Image.composite(face_perturbed, face_source, mask.rotate(angle))
    return result


def apply_shaped_mask_border(face_source, face_perturbed, point):
    border_coef = 0.12
    mask = Image.new("L", face_source.size, 0)
    clearance = [i*border_coef for i in face_source.size]
    print(clearance)
    second_border = tuple([face_source.size[i] - clearance[i] for i in range(2)])
    print(second_border)
    ImageDraw.Draw(mask).ellipse((0, 0) + face_source.size, fill=255)
    ImageDraw.Draw(mask).ellipse((clearance[0], clearance[1]) + second_border, fill=0)
    mask.show()
    angle = -atan((point[1][1] - point[0][1]) / (point[1][0] - point[0][0])) * 180 / pi  # rotate mask by level of eyes
    result = Image.composite(face_perturbed, face_source, mask.rotate(angle))
    return result


def apply_patches(face, lpatch, rpatch, point, face_bounds):
    result = face.copy()
    patch_size = [int(face.size[i]/3) for i in range(2)]
    dist_between_eyes = (point[1][1] - point[0][1])
    angle = -atan(dist_between_eyes / (point[1][0] - point[0][0])) * 180 / pi
    height_correct = int(dist_between_eyes * 0.3)
    lpatch = lpatch.rotate(angle, expand=1).resize(patch_size)
    rpatch = rpatch.rotate(angle, expand=1).resize(patch_size)
    lpbound = [None] * 2
    if fabs(angle) < 20:
        lpbound[0] = int(point[1][0] - face_bounds[0])
    else:
        lpbound[0] = int(point[1][0] - face_bounds[0] - lpatch.width/2)
    lpbound[1] = int(point[1][1] - face_bounds[1] + lpatch.height/4 + height_correct)
    rpbound = [None] * 2
    rpbound[0] = int(point[0][0] - face_bounds[0] - rpatch.width)
    rpbound[1] = int(point[0][1] - face_bounds[1] + rpatch.height/4 - height_correct)
    result.paste(lpatch, lpbound, lpatch)
    result.paste(rpatch, rpbound, rpatch)
    return result


def blur_face(point, face_src):
    """
    :param point: Position of landmarks: Reye, Leye, nose, Rmouth, Lmouth (from face view)
    :param face_src: PIL image - cropped face
    :return: blurred face
    """
    face = face_src.copy()
    if max(face.size) >= 120:
        face = face.filter(ImageFilter.MedianFilter(37))    # ~ 31-49
    else:
        face = face.filter(ImageFilter.MedianFilter(size=21))

    face = apply_shaped_mask(face_src, face, point)
    return face


def process_image(img):
    adv_attack_img = img.copy()
    annotated_faces_img = img.copy()
    blur_img = img.copy()
    boxes, points = extract_faces(img)
    try:
        print(f'Detected {len(boxes)} faces')
    except TypeError:
        print('No faces detected')
        exit()
    for i, (box, point) in enumerate(zip(boxes, points)):
        face_bound = box.astype(int).tolist()
        face = img.crop(face_bound)
        draw_landmarks(box, point, annotated_faces_img, debug_info=True)
        blur_img.paste(blur_face(point, face), face_bound)

        patch = Image.open("noise2d10.png").convert('RGBA')
        patched = apply_patches(face, patch, patch, point, face_bound)
        adv_attack_img.paste(apply_shaped_mask(face, patched, point), face_bound)
    annotated_faces_img.save("annotated_faces.png")
    blur_img.save("blur.png")
    adv_attack_img.save("adv.png")


if __name__ == "__main__":
    image = Image.open('imgs/nure2.jpg')
    # image = Image.open('adv.png')
    process_image(image)
