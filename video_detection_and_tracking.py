from single_image_detection import *
from moviepy.editor import VideoFileClip
import cv2

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
        video_name = 'project_video.mp4'
        output_vid = 'output_project_video.mp4'

        clip1 = VideoFileClip(video_name)
        white_clip = clip1.fl_image(lambda image: single_image_detection(image, svm, scaler))  # NOTE: this function expects color images!!
        white_clip.write_videofile(output_vid, audio=False)
