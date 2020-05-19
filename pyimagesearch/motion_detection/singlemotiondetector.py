# import the necessary packages
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("TkAgg")
import datetime as dt

class SingleMotionDetector:

	def __init__(self, accumWeight = 0.5):  # 0.6 0.4
		# store the accumulated weight factor
		self.accumWeight_bg = accumWeight

		# Main BG
		self.accumWeight_bg_main_initial = 0.75  # 0.45
		self.accumWeight_bg_main = 0.6  # 0.2 0.35
		self.accumWeight_bg_main_slow = 0.35  # 0.07 0.125

		# initialize the background model
		self.bg = None
		self.bg_main = None

		self.initial_bg_frames = 15
		self.counter = 0
		self.prev_thresh = None
		self.updated = False
		self.delta_thresh_bg = 0
		self.updated_area_exist = False
		self.updated_area = 0
		self.sum_thresh_bg = 0
		self.sum_thresh_bg_main = 0
		self.C_previous = []
		self.is_stagnant = False

		self.image_threshold = 10

		# Data collection
		self.contour_areas = []
		self.widths = []
		self.heights = []

		# Features
		self.smart_update = True

		self.plot_mode_on = False

		if self.plot_mode_on:
			self.fig = plt.figure(figsize=(18, 8))
			self.ax1 = self.fig.add_subplot(2, 3, 1)
			self.ax2 = self.fig.add_subplot(2, 3, 2)
			self.ax3 = self.fig.add_subplot(2, 3, 3)
			self.ax4 = self.fig.add_subplot(2, 3, 4)
			self.ax5 = self.fig.add_subplot(2, 3, 5)
			self.ax6 = self.fig.add_subplot(2, 3, 6)

	def update_bg(self, image):

		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight_bg)
		# print("BG updated with weight = " + str(self.accumWeight_bg))


	def update_bg_main(self, image, accumWeight=0.0001):

		# if the background model is None, initialize it
		if self.bg_main is None:
			self.bg_main = image.copy().astype("float")
			return

		cv2.accumulateWeighted(image, self.bg_main, accumWeight)
		# print("Main BG updated with weight = " + str(accumWeight))


	def detect(self, input_image):

		# print("*********************************************************************")
		# print(self.counter)
		# Add blur to avoid specles
		input_image = cv2.GaussianBlur(input_image, (21, 21), 0)

		# Check if bg need initialization
		if self.bg is None:
			self.update_bg(input_image)
		if self.bg_main is None:
			self.update_bg_main(input_image, self.accumWeight_bg_main_initial)

		# BG

		# compute the absolute difference between the background model
		# and the image passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), input_image)
		thresh_bg = cv2.threshold(delta, self.image_threshold, 255, cv2.THRESH_BINARY)[1]

		# Update background after thresh has ben calculated
		self.update_bg(input_image)

		# perform a series of erosions and dilations to remove small
		# blobs
		thresh_bg = cv2.erode(thresh_bg, None, iterations=2)
		thresh_bg = cv2.dilate(thresh_bg, None, iterations=2)

		# Make calculations to determine if main bg should be updated
		# Normalize each pixel to 1
		th1_bg = thresh_bg/255

		# Calculate
		self.sum_thresh_bg = np.sum(th1_bg)

		# BG MAIN

		# Get threshold
		delta2 = cv2.absdiff(self.bg_main.astype("uint8"), input_image)
		thresh_bg_main = cv2.threshold(delta2, self.image_threshold, 255, cv2.THRESH_BINARY)[1]

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
			if len(thresh_bg_main.shape) > 2:
				thresh_bg_main_2D = np.sum(thresh_bg_main.copy(), 2).astype(np.uint8)
			else:
				thresh_bg_main_2D = thresh_bg_main.copy()

			cnts = cv2.findContours(thresh_bg_main_2D,
									cv2.RETR_EXTERNAL,
									cv2.CHAIN_APPROX_SIMPLE)

			cnts = imutils.grab_contours(cnts)

			C = []
			C_small = []
			thr_big = 3000
			thr_too_big = 30000
			thr_small = 200
			# print("Contour areas: ")
			for cnt in cnts:
				# print(cv2.contourArea(cnt))
				# Data collection
				if len(self.contour_areas) < 1000:
					self.contour_areas.append(cv2.contourArea(cnt))

				if cv2.contourArea(cnt) > thr_big: ## and cv2.contourArea(cnt) < thr_too_big:
					C.append(cv2.boundingRect(cnt))
				elif cv2.contourArea(cnt) < thr_small:
					C_small.append(cv2.boundingRect(cnt))

			# Merge overlapping bb
			merge = True
			if merge:
				# duplicate all of them ;)
				l = len(C)
				for r in range(l):
					C.append(C[r])

				# min cluster size = 2, min distance = 0.5:
				C, weights = cv2.groupRectangles(C, 1, 0.75)

			if len(C) > 0:
				# # Check if the bounding box is stagnant (identical to a previous one)
				# if len(self.C_previous) > 0:
				# 	C_tmp = C
				# 	C = []
				# 	for c in C_tmp:
				# 		self.is_stagnant = False
				# 		for pc in self.C_previous:
				# 			if merge:
				# 				if (c == pc).all():
				# 					self.is_stagnant = True
				# 			else:
				# 				if c == pc:
				# 						self.is_stagnant = True
				#
				# 		if not self.is_stagnant:
				# 			C.append(c)
				# 		else:
				# 			print("stagnant")
				#
				# # Save a copy for next time
				# self.C_previous = C

				# Expand boxes with margin
				margin = 100
				for i, c in enumerate(C):

					# Expand box with margin
					x, y, w, h = c

					if len(self.widths) < 1000:
						self.widths.append(w)
						self.heights.append(h)
					# else:
					# 	print("DATA!")

					x = max(0, x - margin)
					y = max(0, y - margin)
					w = min(input_image.shape[1], w + 2 * margin)
					h = min(input_image.shape[0], h + 2 * margin)

					C[i] = (x, y, w, h)

		else:
			C = []

		self.updated_area_exist = False

		# Update background if requirements are met (# self.sum_thresh_bg < 10000 or \)
		if C == [] and \
			self.sum_thresh_bg_main > 500000 or \
			self.counter <= self.initial_bg_frames:
			print("Full pic update" + str(dt.datetime.now()))
			#
			# self.update_bg_main(input_image)  # update background using entire image
			# self.updated = True

			if self.counter < self.initial_bg_frames * 4:
				self.update_bg_main(input_image, self.accumWeight_bg_main_initial)
			else:
				self.update_bg_main(input_image, self.accumWeight_bg_main)

		else:
			# print("Smart pic update")

			# Update outside of BBs

			# initialize all as true (true = update pixel in bg pic)
			idx = th1_bg_main > -1

			# Set all in big rects as False
			for c in C:
				x, y, w, h = c

				idx[y: y+h, x: x+w] = False

			# Set all in small rects as True
			for c in C_small:
				x, y, w, h = c
				idx[y: y+h, x: x+w] = True

			background_pic_bg = self.bg_main.copy()
			if self.counter < self.initial_bg_frames * 4:
				background_pic_bg = input_image
			else:
				background_pic_bg[idx] = input_image[idx]

			# update the background model by accumulating the weighted
			# average
			if self.counter < self.initial_bg_frames * 4:
				self.update_bg_main(background_pic_bg, self.accumWeight_bg_main_initial)
			else:
				self.update_bg_main(background_pic_bg, self.accumWeight_bg_main)

			if self.plot_mode_on:
				background_pic_bg = self.bg_main.copy()*0
				background_pic_bg[idx] = input_image[idx]

				self.ax5.clear()
				self.ax5.imshow(background_pic_bg.astype(int))
				self.ax5.set_title("Main BG fast update input")

			self.updated_area = th1_bg_main.copy()*0
			self.updated_area[idx] = 1
			self.updated_area_exist = True


			## Update inside BBs:

			# initialize all as false (true = update pixel in bg pic)
			idx = th1_bg_main == -1

			# Set all in big rects as True
			for c in C:
				x, y, w, h = c

				idx[y: y+h, x: x+w] = True

			background_pic_main = self.bg_main.copy()
			background_pic_main[idx] = input_image[idx]
			if self.counter > self.initial_bg_frames * 4:
				self.update_bg_main(background_pic_main, self.accumWeight_bg_main_slow)

			self.updated = False

			if self.plot_mode_on:

				self.ax1.clear()
				self.ax1.imshow(input_image.astype(int))
				self.ax1.set_title("Input image")

				self.ax2.clear()
				self.ax2.imshow(self.bg_main.astype(int))
				self.ax2.set_title("Main BG")

				self.ax3.clear()
				self.ax3.imshow(self.bg.astype(int))
				self.ax3.set_title("BG")

				background_pic_main_x = self.bg_main*0
				background_pic_main_x[idx] = input_image[idx]
				self.ax4.clear()
				self.ax4.imshow(background_pic_main_x.astype(int))
				self.ax4.set_title("BG Main Slow Input")

				self.ax6.clear()
				self.ax6.imshow(input_image.astype(int))
				self.ax6.set_title("BG update input")

				plt.draw()
				plt.pause(0.002)
				plt.ioff()
				plt.show()


		if C == []:
			return None, None
		else:
			return thresh_bg_main, C