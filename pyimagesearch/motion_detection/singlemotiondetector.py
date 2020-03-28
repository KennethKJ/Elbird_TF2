# import the necessary packages
import numpy as np
import imutils
import cv2


class SingleMotionDetector:

	def __init__(self, accumWeight = 0.5):  # 0.4
		# store the accumulated weight factor
		self.accumWeight_bg = accumWeight
		self.accumWeight_bg_main = 0.5

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
		self.C_previous = []

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

	def detect(self, image, tVal = 10):

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
		self.sum_thresh_bg = np.sum(th1_bg)


		# BG MAIN

		# Get threshold
		delta2 = cv2.absdiff(self.bg_main.astype("uint8"), image)
		thresh_bg_main = cv2.threshold(delta2, tVal, 255, cv2.THRESH_BINARY)[1]

		# perform a series of erosions and dilations to remove small
		# blobs
		thresh_bg_main = cv2.erode(thresh_bg_main, None, iterations=2)
		thresh_bg_main = cv2.dilate(thresh_bg_main, None, iterations=2)

		th1_bg_main = thresh_bg_main/255
		self.sum_thresh_bg_main = np.sum(th1_bg_main)

		self.counter = self.counter + 1

		if self.counter > self.initial_bg_frames:

			# find contours in the thresholded image and initialize the
			# minimum and maximum bounding box regions for motion
			if len(thresh_bg_main.shape)> 2:
				thresh_bg_main_2D = np.sum(thresh_bg_main.copy(), 2).astype(np.uint8)
			else:
				thresh_bg_main_2D = thresh_bg_main.copy()

			cnts = cv2.findContours(thresh_bg_main_2D,
									cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)

			cnts = imutils.grab_contours(cnts)

			C = []
			C_small = []
			thrsh = 2000
			for cnt in cnts:
				if cv2.contourArea(cnt) > thrsh:
					C.append(cv2.boundingRect(cnt))
				elif cv2.contourArea(cnt) < thrsh:
					C_small.append(cv2.boundingRect(cnt))



			if C != []:


				# #  Check if the bounding box is stagnant (identical to a previous one)

				if self.C_previous != []:
					C_tmp = C
					C = []
					for c in C_tmp:
						is_stagnant = False
						for pc in self.C_previous:
							if c == pc:
								is_stagnant = True

						if not is_stagnant:
							C.append(c)

				# Save a copy for next time
				self.C_previous = C



				# Expand boxes with margin
				margin = 0
				for i, c in enumerate(C):

					# Expand box with margin
					x, y, w, h = c

					x = max(0, x - margin)
					y = max(0, y - margin)
					w = max(0, w + 2*margin)
					h = max(0, h + 2*margin)

					C[i] = (x, y, w, h)





		else:
			C = None


		smart_update =  False

		continuous_update = True

		# Update background if requirements are met
		if self.sum_thresh_bg < 2000 or \
			self.sum_thresh_bg_main > 500000 or \
			self.counter <= self.initial_bg_frames or \
			continuous_update:

			self.update_bg_main(image)  # update background using entire image
			self.updated = True

		else:

			if smart_update:

				idx = th1_bg_main > -1  # initialize all as true (true = update pixel in bg pic)

				# Set all in big rects as False
				for c in C:
					x, y, w, h = c

					idx[y: y+h, x: x+w] = False

				# Set all in small rects as True
				for c in C_small:
					x, y, w, h = c

					idx[y: y+h, x: x+w] = True

				# Initialize pik
				pik = self.bg_main

				pik[idx] = image[idx]

				self.update_bg_main(pik)



			self.updated = False

		if C is None:
			return None
		else:
			return thresh_bg_main, C