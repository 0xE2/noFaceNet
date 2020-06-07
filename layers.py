import numpy as np


class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.rand(num_filters, 3, 3) / 9    # Xavier initialization

    @staticmethod
    def iterate_regions(image, padding=0):
        """
        Generates all possible 3x3 image regions using valid padding
        :param image: 2d np array
        """
        if padding:
            image_ = np.zeros((image.shape[0] + 2*padding, image.shape[1] + 2*padding))
            image_[padding:image.shape[0]+padding, padding:image.shape[1]+padding] = image
            image = image_
            del image_
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[i:(i + 3), j:(j + 3)]
                yield img_region, i, j

    def forward(self, input, padding=0):
        """
        Performs a forward pass of the conv layer using the given input
        :param input: 2d np array
        :return: 3d np array with dimensions (h, w, num_filters)
        """
        self.input = input
        try:
            h, w = input.shape
            output = np.zeros((h - 2, w - 2, self.num_filters))

            for img_region, i, j in self.iterate_regions(input, padding):
                output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))
            return output
        except ValueError:
            h, w, d = input.shape
            out = np.zeros((h - 2, w - 2, d * self.num_filters))
            # for i, inp in np.rollaxis(input, 2):
            for depth in range(d):
                output = np.zeros((h - 2, w - 2, self.num_filters))

                for img_region, i, j in self.iterate_regions(input[:, :, depth], padding):
                    output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))
                for a in range(self.num_filters):
                    out[:, :, depth+a] = output[:, :, a]
            return out

    def backprop(self, grad, learn_rate):
        """
        Performs a backward pass of the conv layer
        :param grad: loss gradient for this layer's outputs
        :param learn_rate: float
        :return: loss gradient for this layer's inputs
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        try:
            _, _, d = self.input.shape
            for depth in range(d):
                for img_region, i, j in self.iterate_regions(self.input[:, :, depth]):
                    for f in range(self.num_filters):
                        d_L_d_filters[f] += grad[i, j, f] * img_region
        except ValueError:
            for img_region, i, j in self.iterate_regions(self.input):
                for f in range(self.num_filters):
                    d_L_d_filters[f] += grad[i, j, f] * img_region

        d_L_d_input = np.zeros(self.input.shape)
        for f in range(self.num_filters):
            kernel_rot180 = np.rot90(self.filters[f], 2)
            for img_region, i, j in self.iterate_regions(grad[f], padding=1):
                try:
                    d_L_d_input[i, j, f] += np.sum(kernel_rot180 * img_region)
                except IndexError:
                    d_L_d_input[i, j] += np.sum(kernel_rot180 * img_region)

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        return d_L_d_input


class MaxPool:
    @staticmethod
    def iterate_regions(image):
        """
        :param image: 2d np array
        :return: non-overlapping 2x2 image regions to pool over
        """
        h, w, d = image.shape
        h_new = h // 2
        w_new = w // 2

        for i in range(h_new):
            for j in range(w_new):
                img_region = image[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]
                yield img_region, i, j

    def forward(self, input):
        """
        Performs a forward pass of the maxpool layer using the given input
        :param input: 3d np array with dim: (h, w, num_filters)
        :return: 3d np array with dim: (h / 2, w / 2, num_filters)
        """
        self.input = input
        h, w, d = input.shape
        output = np.zeros((h // 2, w // 2, d))

        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(img_region, axis=(0, 1))
        return output

    def backprop(self, grad):
        """
        Performs a backward pass of the maxpool layer
        :param grad: loss gradient for this layer's outputs
        :return: loss gradient for this layer's inputs
        """
        d_L_d_input = np.zeros(self.input.shape)
        for img_region, i, j in self.iterate_regions(self.input):
            h, w, d = img_region.shape
            amax = np.amax(img_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for d2 in range(d):
                        # If this pixel was the max value, copy the gradient to it
                        if img_region[i2, j2, d2] == amax[d2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, d2] = grad[i, j, d2]

        return d_L_d_input


class Dense:
    """
    A standard fully-connected layer
    Possible activations: relu, tanh, sigmoid, softmax (not tested)
    """
    def __init__(self, input_len, out_len, activation="relu"):
        """
        :param input_len: input dim
        :param out_len: output dim
        :param activation: relu, softmax, tanh
        """
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.rand(input_len, out_len) / input_len
        self.biases = np.zeros(out_len)
        self.activation = activation

    def forward(self, input):
        """
        Performs a forward pass using the given input
        :param input: np array of any dimension
        :return: output 1d np array
        """
        self.input_shape = input.shape

        input = input.flatten()
        self.input = input
        input_len, out_len = self.weights.shape

        result = np.dot(input, self.weights) + self.biases
        self.result = result

        if self.activation == "relu":
            out = np.maximum(0, result)
            # out = result * (result > 0)   # One more implementation
        elif self.activation == "softmax":
            exp = np.exp(result)
            out = exp / np.sum(exp, axis=0)
        elif self.activation == "tanh":
            out = np.tanh(result)
        elif self.activation == "sigmoid":
            out = 1/(1 + np.exp(-result))
        return out

    def backprop(self, grad, learn_rate):
        """
        Performs a backward pass of the fully connected layer
        :param grad: loss gradient for layer outputs
        :param learn_rate: float
        :return: loss gradient for layer's inputs
        """
        if self.activation == "relu":
            d_out_d_z = self.result >= 0
        elif self.activation == "softmax":
            i = grad.nonzero()
            z_exp = np.exp(self.result)
            S = np.sum(z_exp)
            # Gradients of out[i] against totals
            d_out_d_z = -z_exp[i] * z_exp / (S ** 2)
            d_out_d_z[i] = z_exp[i] * (S - z_exp[i]) / (S ** 2)
        elif self.activation == "tanh":
            d_out_d_z = np.ones_like(self.result) - np.tanh(self.result) ** 2
        elif self.activation == "sigmoid":
            d_out_d_z = 1 / (1 + np.exp(-self.result)) * (1 - 1/(1 + np.exp(-self.result)))
        d_z_d_w = self.input
        d_z_d_b = 1
        d_z_d_inputs = self.weights

        d_L_d_z = grad.flatten() * d_out_d_z
        # Gradients of loss against weights/biases/input
        d_L_d_w = d_z_d_w[np.newaxis].T @ d_L_d_z[np.newaxis]
        d_L_d_b = d_L_d_z * d_z_d_b
        d_L_d_inputs = d_z_d_inputs @ d_L_d_z

        # Update weights / biases
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b
        return d_L_d_inputs.reshape(self.input_shape)


if __name__ == "__main__":
    c = Conv3x3(4)
    mpool = MaxPool()
    dense = Dense(10, 10)
