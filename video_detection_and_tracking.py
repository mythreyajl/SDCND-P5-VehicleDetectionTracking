from single_image_detection import *
from moviepy.editor import VideoFileClip
import cv2
import collections

count = 0


def area(bb):
    w = np.absolute(bb[0][0] - bb[1][0])
    h = np.absolute(bb[0][1] - bb[1][1])
    return w*h


def union_bb(bb1, bb2):
    left = min(bb1[0][0], bb2[0][0])
    right = max(bb1[1][0], bb2[1][0])
    top = min(bb1[0][1], bb2[0][1])
    bottom = max(bb1[1][1], bb2[1][1])
    return (left - right) * (top - bottom)


def detect_disjoint(bb1, bb2):
    union = union_bb(bb1, bb2)
    total = area(bb1) + area(bb2)
    if union > total:
        return True
    return False


def intersection(bb1, bb2):
    if detect_disjoint(bb1, bb2):
        return -1

    left = max(bb1[0][0], bb2[0][0])
    right = min(bb1[1][0], bb2[1][0])
    top = max(bb1[0][1], bb2[0][1])
    bottom = min(bb1[1][1], bb2[1][1])
    return (left - right) * (top - bottom)


def intersection_of_union(bb1, bb2):
    return intersection(bb1, bb2) / union_bb(bb1, bb2)


class Track:

    def __init__(self, bb):
        self.frames = collections.deque(maxlen=10)
        self.add_frame(bb)

    def calculate_rep(self):
        if len(self.frames) == 1:
            self.rep_frame = self.frames[0]
        else:
            top = []
            bottom = []
            left = []
            right = []
            for frame in self.frames:
                top.append(frame[0][1])
                bottom.append(frame[1][1])
                left.append(frame[0][0])
                right.append(frame[1][0])
            self.rep_frame = ((np.int(np.median(left)), np.int(np.median(top))),
                              (np.int(np.median(right)), np.int(np.median(bottom))))

    def add_frame(self, bb):
        self.frames.append(bb)
        self.calculate_rep()

    frames = None
    rep_frame = None
    last_frame = None


class TrackMgr:

    def __init__(self):
        self.tracks = collections.deque()

    def new_track(self, bb):
        track = Track()
        track.frames.append(bb)
        track.last_frame = 0
        self.tracks.append(track)
        return

    def add_frame(self, bb, count):
        max_iou = 0.0001
        cnt = -1
        track_id = cnt
        for track in self.tracks:
            cnt += 1
            if detect_disjoint(track.rep_frame, bb):
                continue
            iou = intersection_of_union(track.rep_frame, bb)
            if iou > max_iou:
                max_iou = iou
                track_id = cnt

        if max_iou > 0.3:
            self.tracks[track_id].last_frame = count
            self.tracks[track_id].add_frame(bb)

        if track_id == -1:
            new_track = Track(bb)
            new_track.last_frame = count
            self.tracks.append(new_track)

    def get_bbs(self, count):
        bbs = []
        for track in self.tracks:
            if track.last_frame < count - 10:
                continue
            if len(track.frames) > 1:
                bbs.append(track.rep_frame)

        return bbs

    tracks = None


def draw_stable_bboxes(img, labels, count):

    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        w = np.absolute(bbox[0][0]-bbox[1][0])
        h = np.absolute(bbox[0][1]-bbox[1][1])
        ratio = max(w, h) / min(w, h)
        if ratio > 8:
           continue
        tracks.add_frame(bbox, count)

    final_bbs = tracks.get_bbs(count)
    for bbox in final_bbs:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 6)

    return img


def single_image_detection(image, model, scaler, save=False, video_mode=True):
    global count
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    windows = find_cars(np.copy(image), ystart=380, ystop=550, xstart=600, scale=1, model=model, scaler=scaler,
                        orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

    windows += find_cars(np.copy(image), ystart=400, ystop=656, xstart=600, scale=1.5, model=model, scaler=scaler,
                         orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    labels = heat_map(heat, windows, ratio=0.25)

    # Display results
    overlaid = draw_stable_bboxes(np.copy(image), labels, count)
    if save:
        cv2.imwrite("./output_images/image_" + str(count) + ".jpg", overlaid)
        count += 1
    # cv2.imshow('ol', overlaid)
    # cv2.waitKey(0)

    overlaid = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

    return overlaid


def parse_args():

    parser = argparse.ArgumentParser(description="Detect lane markers on video")
    parser.add_argument('-v', '--video', dest="video", help='Input file to be processed')
    parser.add_argument('-s', '--standalone', dest="standalone", help='Run pipeline on a single example image')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    train = pickle.load(open('train.p', 'rb'))  #
    svm = train["model"]
    X_test = train["X_test"]
    y_test = train["y_test"]
    scaler = train["X_scaler"]

    if args.standalone:
        files = glob("./test_images/test*.jpg")

        for filename in files:
            image = cv2.imread(filename)

            windows = find_cars(np.copy(image), ystart=380, ystop=656, xstart=600, scale=1, model=svm, scaler=scaler,
                                orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

            windows += find_cars(np.copy(image), ystart=400, ystop=600, xstart=600, scale=1.5, model=svm, scaler=scaler,
                                 orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

            heat = np.zeros_like(image[:, :, 0]).astype(np.float)
            labels = heat_map(heat, windows, 0.2)

            # Display results
            overlaid = draw_labeled_bboxes(np.copy(image), labels)
            cv2.imshow('boxes', overlaid)
            cv2.waitKey(0)

    else:
        tracks = TrackMgr()

        video_name = 'project_video.mp4'
        output_vid = 'output_project_video_tracking.mp4'

        clip1 = VideoFileClip(video_name)
        white_clip = clip1.fl_image(lambda image: single_image_detection(image, svm, scaler, True, True))  # NOTE: this function expects color images!!
        white_clip.write_videofile(output_vid, audio=False)
