import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import Qt
from PIL import Image, ImageQt
import utils
import noFaceNet  # process_photo, forward_step
import sys

MAX_HEIGHT = 700


def to_PIL_img(pixmap):
    """
    Auxiliary func. Converts QPixel to PIL Image
    :param pixmap: QPixel
    :return: PIL Image
    """
    import io
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    pixmap.save(buffer, "PNG")
    return Image.open(io.BytesIO(buffer.data()))


class Face:
    def __init__(self, box, points, color=(255, 255, 255), colorm=(15, 240, 252)):
        self.box = box
        self.points = points
        self.color = color
        self.colorm = colorm
        self.static = True
        self.dest = None

    def tick(self):
        if self.dest is not None:
            delta_space = [abs(self.box[i] - self.dest[i]) for i in range(4)]
            if min(delta_space) < 2:    # Del point from comparison if it enough close to its destination
                if delta_space.index(min(delta_space)) <= 1:
                    delta_space = delta_space[2:]
                else:
                    delta_space = delta_space[:2]
            if 0 < sum(delta_space) <= 25:
                self.box = self.dest
                self.dest = None
                return
            if sum(delta_space):
                eval_ = lambda delta, a, b: int(
                    delta * (self.dest[a] - self.box[a]) / (self.dest[b] - self.box[b]) + self.box[a])
                dlt = 1.8
                if not delta_space.index(min(delta_space)) % 2:  # Delta X is the smallest, so choosing X as "master" in
                    # frame movement because it would be moving faster
                    dx1 = dlt if self.dest[0] > self.box[0] else -dlt
                    dx2 = dlt if self.dest[2] > self.box[2] else -dlt
                    if self.dest[0] != self.box[0]:
                        y1 = eval_(dx1, 1, 0)
                    else:
                        y1 = self.box[1]
                    if self.dest[2] != self.box[2]:
                        y2 = eval_(dx2, 3, 2)
                    else:
                        y2 = self.box[3]
                    self.box[0] += dx1
                    self.box[2] += dx2
                    self.box[1] = y1
                    self.box[3] = y2
                else:
                    dy1 = dlt if self.dest[1] > self.box[1] else -dlt
                    dy2 = dlt if self.dest[3] > self.box[3] else -dlt
                    if self.dest[1] != self.box[1]:
                        x1 = eval_(dy1, 0, 1)
                    else:
                        x1 = self.box[0]
                    if self.dest[3] != self.box[3]:
                        x2 = eval_(dy2, 2, 3)
                    else:
                        x2 = self.box[2]
                    self.box[0] = x1
                    self.box[2] = x2
                    self.box[1] += dy1
                    self.box[3] += dy2
            else:
                self.dest = None

    def draw(self, img):
        color = self.color if self.static else self.colorm
        utils.draw_landmarks(self.box, self.points, img, color=color)


class Load(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.is_active = False
        self.set_img('deny.png')  # TODO: use resource instead
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAcceptDrops(True)

    def change_state(self):
        self.is_active = not self.is_active
        self.set_img('accept.png' if self.is_active else 'deny.png')

    def set_img(self, img):
        pix = QtGui.QPixmap(img)
        # self.h = pix.height()
        # self.w = pix.width()
        self.setPixmap(pix)

    def tick(self):
        pass

    def dragEnterEvent(self, event):
        # mime_data = event.mimeData()
        # print(event.mimeData().hasText(), event.mimeData().hasHtml(),
        #       event.mimeData().hasUrls(), event.mimeData().hasImage())
        self.change_state()
        event.acceptProposedAction()

    # def dragMoveEvent(self, event):   # TODO: make animation for Load scene?
    #     event.acceptProposedAction()

    def dropEvent(self, event):
        # files = [unicode(u.toLocalFile()) for u in event.mimeData().urls()] # TODO: test this implementation
        mime_data = event.mimeData()
        if mime_data.hasImage():
            image = QtGui.QImage(mime_data.imageData())
        elif mime_data.hasUrls():
            image = mime_data.text()[8:]
        self.parent.setWidget(Processing(image, parent=self.parent))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.change_state()
        event.accept()


class Processing(QtWidgets.QLabel):
    def __init__(self, photo, parent=None):
        super().__init__()
        self.parent = parent
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.faces = []
        self.slides = []
        self.pointer = 0
        self.set_img(photo, stack=True)
        self.parent.center_window()  # TODO: won't work here. Where to move this?

        self.bind('Ctrl+D', self.detect)
        self.bind('Ctrl+B', self.blur)
        self.bind(Qt.Qt.Key_B, self.blur_selected)
        self.bind('Ctrl+P', self.patch)
        self.bind('Ctrl+Q', lambda: QtWidgets.QApplication.quit())
        self.bind(Qt.Qt.Key_Right, self.key_right)
        self.bind(Qt.Qt.Key_Left, self.key_left)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.Qt.BackButton:  # slide back
            if len(self.slides) > 1:
                self.stack()

    def bind(self, key_sequence, action):
        QtWidgets.QShortcut(QtGui.QKeySequence(key_sequence), self).activated.connect(action)

    def detect(self):
        boxes, points = utils.extract_faces(self.slides[-1])

        self.faces = []
        if boxes is None:
            return

        # Sort faces by first X coord
        permutation = boxes[:, 0].argsort()
        boxes = boxes[permutation].astype(np.int16)
        points = points[permutation].astype(np.int16)

        for i, face in enumerate(zip(boxes, points)):
            self.faces.append(Face(face[0], face[1]))
        self.faces.append(Face(self.faces[0].box.copy(), self.faces[0].points.copy()))
        self.pointer = 0

    def blur(self):
        self.stack(self.slides[-1])
        for i in range(len(self.faces) - 1):
            face = self.slides[-1].crop(self.faces[i].box)
            self.slides[-1].paste(utils.blur_face(self.faces[i].points, face), self.faces[i].box)

    def blur_selected(self):
        self.stack(self.slides[-1])
        i = self.pointer
        face = self.slides[-1].crop(self.faces[i].box)
        self.slides[-1].paste(utils.blur_face(self.faces[i].points, face), self.faces[i].box)

    def patch(self):
        self.stack(self.slides[-1])
        self.slides[-1] = noFaceNet.process_photo(self.slides[-1])

    def patch_selected(self):
        # self.stack(self.slides[-1]) # TODO: some refactoring in noFaceNet needed to implement this
        pass

    def set_traced_face(self):
        self.faces[-1].static = False
        self.faces[-1].dest = self.faces[self.pointer].box.copy()
        self.faces[-1].points = self.faces[self.pointer].points.copy()

    def key_right(self):
        self.pointer += 1
        if self.pointer == len(self.faces) - 1:
            self.pointer = 0
        self.set_traced_face()

    def key_left(self):
        self.pointer -= 1
        if self.pointer == -1:
            self.pointer = len(self.faces) - 2
        self.set_traced_face()

    def set_img(self, img, stack=False):
        if isinstance(img, (str, QtGui.QImage)):
            pix = QtGui.QPixmap(img)
        elif isinstance(img, Image.Image):
            img = ImageQt.ImageQt(img)
            pix = QtGui.QPixmap.fromImage(img)
        if stack:
            self.stack(pix)

        pix = pix.scaled(int(MAX_HEIGHT / pix.height() * pix.width()), MAX_HEIGHT, QtCore.Qt.IgnoreAspectRatio)
        self.setPixmap(pix)

    def stack(self, image=None):
        if image:
            if len(self.slides) == 5:
                self.slides = [self.slides[i] for i in range(1, len(self.slides))]
            if isinstance(image, QtGui.QPixmap):
                self.slides.append(to_PIL_img(image))
            else:  # PIL image
                self.slides.append(image.copy())
        else:
            self.slides.pop(-1)

    def tick(self):
        frame = self.slides[-1].copy()
        for face in self.faces:
            face.tick()
            face.draw(frame)
        self.set_img(frame)
        self.parent.center_window()


class Wrapper(QtWidgets.QMainWindow):
    def __init__(self, screen_resolution):
        super().__init__()
        self.setWindowTitle("NoFaceNet")
        self.setStyleSheet("background-color: gray;")
        self.setGeometry(700, 200, 100, 100)
        self.setMaximumSize(screen_resolution)

        self.scene = Load(parent=self)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.lay = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, central_widget)
        self.setWidget(self.scene)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(16)  # ~ 62.5 FPS max
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        self.show()

    def setWidget(self, widget):
        try:
            self.lay.itemAt(0).widget().setParent(None)
        except AttributeError:
            pass
        self.scene = widget
        # self.resize(self.scene.w, self.scene.h)
        self.lay.insertWidget(0, widget)

    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def tick(self):
        self.scene.tick()


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    win = Wrapper(screen_resolution.size())
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
