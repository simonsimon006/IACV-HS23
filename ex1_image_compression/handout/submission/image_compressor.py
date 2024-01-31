import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.integrate import quad
from typing import Iterable


def lambd(ß):
	return np.sqrt(2 * (ß + 1) + 8 * ß / ((ß + 1) + np.sqrt(ß**2 + 14 * ß + 1)))


def intform(t, ß):
	return np.sqrt(
	    ((1 + np.sqrt(ß))**2 - t) * (t - (1 - np.sqrt(ß)**2))) / (2 * np.pi * t)


def intgl(u, ß):
	y = quad(intform, (1 - ß)**2, u, args=(ß, ))
	# The result is in y[0], y[1] holds e.g. the error.

	return y[0] - 0.5


def thresh(n: int, m: int, sigmas: Iterable[float]):
	ß = n / m
	u = fsolve(func=intgl, x0=(300, ), args=(ß, ), maxfev=400)
	w = lambd(ß) / u
	sig = np.median(sigmas)
	return (w * sig)[0]


def denoise(ten: np.ndarray):
	# No full_matrices so the matmul simply works without padding
	# as the 0 svalues are omitted by the function.
	U, E, V = np.linalg.svd(ten, full_matrices=False)

	t = thresh(ten.shape[0], ten.shape[1], E)
	Ec = np.zeros_like(E)
	indexs = E > t
	Ec[indexs] = E[indexs]

	X = (U * Ec) @ V
	return X, E, Ec, t


class ImageCompressor:
	"""
	  This class is responsible to
		  1. Learn the codebook given the training images
		  2. Compress an input image using the learnt codebook
	"""

	def __init__(self):
		"""
		Feel free to add any number of parameters here.
		But be sure to set default values. Those will be used on the evaluation server
		"""

		# Here you can set some parameters of your algorithm, e.g.
		self.dtype = np.float32
		self.codebook = np.array([])
		self.threshold = None

	def get_codebook(self):
		""" Codebook contains all information needed for compression/reconstruction """

		return self.codebook.astype(self.dtype)

	def train(self, train_images):
		"""
		Training phase of your algorithm - e.g. here you can perform PCA on training data
		
		Args:
			train_images  ... A list of NumPy arrays.
							  Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
		"""
		N = len(train_images)

		images_flat = [image.flat for image in train_images]
		image_matrix = np.stack(images_flat).T.astype(self.dtype)

		mean = np.mean(image_matrix, axis=1)

		for x in range(image_matrix.shape[1]):
			image_matrix[:, x] -= mean

		U, E, V = np.linalg.svd(image_matrix, full_matrices=False)

		t = thresh(image_matrix.shape[0], image_matrix.shape[1], E) + E.mean()
		t *= 1.12
		made_it = E >= t

		U_th = U[:, made_it]

		self.threshold = t
		temp = np.zeros((U_th.shape[0], U_th.shape[1] + 1))
		temp[:, :-1] = U_th
		temp[:, -1] = mean
		self.codebook = temp

	def compress(self, test_image):
		""" Given an array of shape H x W x C return compressed code """

		flat_image = test_image.flat - self.codebook[:, -1]
		values = self.codebook[:, :-1].T @ flat_image
		return values.astype(self.dtype)


class ImageReconstructor:
	""" This class is used on the server to reconstruct images """

	def __init__(self, codebook):
		""" The only information this class may receive is the codebook """
		self.codebook = codebook

	def reconstruct(self, test_code):
		""" Given a compressed code of shape K, reconstruct the original image """
		image = self.codebook[:, :-1] @ test_code + self.codebook[:, -1]

		dist = image.max() - image.min()

		image += np.abs(image.min())
		image /= image.max()

		image *= 255

		image = (image.reshape((96, 96, 3))).astype("uint8")

		for x in range(96):
			for y in range(96):
				if (image[x, y, :] > 160).all():
					image[x, y] = [255, 255, 255]
				elif image[x, y, 0] > 150:
					image[x, y, :] = [212, 50, 48]
				elif (image[x, y, :] < 50).all():
					image[x, y, :] = [0, 0, 0]
				elif (image[x, y, 1] > 120 and image[x, y, 0] < 100):
					image[x, y, :] = [61, 137, 68]
		return image
