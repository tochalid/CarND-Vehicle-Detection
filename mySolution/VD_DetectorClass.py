from collections import deque

from scipy.ndimage.measurements import label

from mySolution.VD_functions import *

import matplotlib.pyplot as plt
import datetime
import time


class Detector:
    def __init__(self, svc, scaler):

        self.Scaler = scaler
        self.SVC = svc
        self.window_scale = (1.0)  # scale window
        self.x_start_stop = [[0, 1280]]  # x span of the window of the scanner
        self.y_start_stop = [[0, 720]]  # y span of the window of the scanner
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.threshold = 1  # detection threshold, corresponds with hot-count required for a pixel to be labeled as car pixel
        self.smoothing_limit = 1  # Number of video frames adding heat before applying threshold
        self.heatmaps_list = deque(maxlen=self.smoothing_limit)
        self.color_space = 'RGB'

    def integrate_heatmap(self, heatmaps_list, thresh):
        smooth_heatmap = 0
        # Integrate hot pixel to a maximum of smooth_limit
        for i in range(len(heatmaps_list)):
            smooth_heatmap += heatmaps_list[i]
        # Apply threshold and return updated map
        return apply_threshold(smooth_heatmap, thresh)

    def vehicle_detection(self, img, DIR='../output_images/video/'):

        t0 = time.time()
        detection_windows = []

        for i, scale in enumerate(self.window_scale):
            detection_windows.extend(
                find_cars(img, self.x_start_stop[i][0], self.x_start_stop[i][1], self.y_start_stop[i][0],
                          self.y_start_stop[i][1], scale, self.SVC,
                          self.Scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size,
                          self.hist_bins, DIR=DIR, color_space=self.color_space))

        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        heatmap = add_heat(heatmap, detection_windows)

        # smoothening with threshold over entire list
        self.heatmaps_list.append(heatmap)
        smooth_heatmap = self.integrate_heatmap(self.heatmaps_list, self.threshold)
        labels = label(smooth_heatmap)

        ##Draw labels or detection windows
        final_img = draw_labeled_bboxes(img, labels)

        if False:  # just a switch to suppress image output if not needed
            # Section for saving images for the project documentation
            # timestamp for saving to disk
            d = datetime.datetime.now().strftime('%Y-%m-%d_w%W_%H:%M:%S.%f')

            # Image that shows all detected windows based on the output list of the classifier
            img_onwin = draw_boxes(img, detection_windows)
            plt.imshow(img_onwin)
            plt.savefig(DIR + 'on_win/on_windows_' + d + '.png')

            # Image that shows the detected windows with integrated filled rectangels
            blank = np.zeros_like(img)
            img_integ = draw_boxes(blank, detection_windows, color=(125, 125, 125), thick=-1)  # -1 fills the rectangel
            plt.imshow(img_integ, cmap='gray')
            plt.savefig(DIR + 'integ/integ_' + d + '.png')

            # Image that shows the final result, the labeled car
            plt.imshow(final_img)
            plt.savefig(DIR + 'labeled_bboxes/lbboxes_' + d + '.png')

            # Image that shows the heatmap
            heatmap = np.clip(smooth_heatmap, 0, 255)
            plt.imshow(heatmap, cmap='hot')
            plt.savefig(DIR + 'heatmap/heatmap_' + d + '.png')

            # Five images in one column visualizing the pipeline result and its steps, used for the project documentation
            fig = plt.figure(figsize=(4, 20))
            plt.subplot(511)
            plt.imshow(img)
            plt.title('Original')
            plt.subplot(512)
            plt.imshow(img_onwin)
            plt.title('Car Detections')
            plt.subplot(513)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.subplot(514)
            plt.imshow(img_integ)
            plt.title('Labeled Detections')
            plt.subplot(515)
            plt.imshow(final_img)
            plt.title('Labeled Car')
            fig.tight_layout()
            plt.savefig(DIR + 'lbb_h_compare/llb_h_compare_' + d + '.png')
            plt.close('all')

        # returning the labeled car image
        return final_img
