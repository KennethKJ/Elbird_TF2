# import the necessary packages
import numpy as np
import imutils
import cv2

class SingleMotionDetector:
	def __init__(self, accumWeight= 0.4): # 0.4
		# store the accumulated weight factor
		self.accumWeight_bg = accumWeight + 0.15
		self.accumWeight_bg_main = accumWeight

		# initialize the background model
		self.bg = None
		self.bg_main = None

		self.initial_bg_frames = 15
		self.counter = 0
		self.prev_thresh = None
		self.updated = False
		self.delta_thresh_bg = 0
		self.sum_thresh_bg = 0
		self.sum_thresh_bg_main = 0

	def update_bg(self, image):

		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight_bg)

	def update_bg_main(self, image):

		# if the background model is None, initialize it
		if self.bg_main is None:
			self.bg_main = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg_main, self.accumWeight_bg_main)

	def detect(self, image, tVal = 20):

		# Add blur to avoid specles
		image = cv2.GaussianBlur(image, (21, 21), 0)


		# BG

		# compute the absolute difference between the background model
		# and the image passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh_bg = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]


		# perform a series of erosions and dilations to remove small
		# blobs
		thresh_bg = cv2.erode(thresh_bg, None, iterations=2)
		thresh_bg = cv2.dilate(thresh_bg, None, iterations=2)

		# Update background after thresh has ben calculated
		self.update_bg(image)

		# Make calculations to determine if main bg should be updated
		# Normalize each pixel to 1
		th1_bg = thresh_bg/255

		# Calculate
		self.sum_thresh_bg = sum(sum(th1_bg))


		# BG MAIN

		# Get threshold
		delta = cv2.absdiff(self.bg_main.astype("uint8"), image)
		thresh_bg_main = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

		# perform a series of erosions and dilations to remove small
		# blobs
		thresh_bg_main = cv2.erode(thresh_bg_main, None, iterations=2)
		thresh_bg_main = cv2.dilate(thresh_bg_main, None, iterations=2)

		th1_bg_main = thresh_bg_main/255
		self.sum_thresh_bg_main = sum(sum(th1_bg_main))

		self.initial_bg_frames = 15
		self.counter = self.counter + 1

		# Update background if requirements met
		if self.sum_thresh_bg < 2000 or self.sum_thresh_bg_main > 500000 or self.counter <= self.initial_bg_frames:
			self.update_bg_main(image)
			self.updated = True
		else:
			self.updated = False

		if self.counter > self.initial_bg_frames:
			# find contours in the thresholded image and initialize the
			# minimum and maximum bounding box regions for motion
			cnts = cv2.findContours(thresh_bg_main.copy(),
									cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)

			cnts = imutils.grab_contours(cnts)

			# for cnt in cnts:
			# 	print(cv2.contourArea(cnt))

			cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 5000]
			# (minX, minY) = (np.inf, np.inf)
			# (maxX, maxY) = (-np.inf, -np.inf)

			# if no contours were found, return None
			# if len(cnts) == 0:
			# 	return None

			C = [cv2.boundingRect(c) for c in cnts]

		else:
			return None

		# # otherwise, loop over the contours
		# for c in cnts:
		# 	# compute the bounding box of the contour and use it to
		# 	# update the minimum and maximum bounding box regions
		# 	(x, y, w, h) = cv2.boundingRect(c)
		# 	(minX, minY) = (min(minX, x), min(minY, y))
		# 	(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		# otherwise, return a tuple of the thresholded image along
		# with bounding box
		# return (thresh, (minX, minY, maxX, maxY))
		return (thresh_bg_main, C)