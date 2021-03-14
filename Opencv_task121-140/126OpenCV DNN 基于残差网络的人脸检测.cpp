#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;
String model_bin = "models/face_detector/opencv_face_detector_uint8.pb";
String config_text = "models/face_detector/opencv_face_detector.pbtxt";

int main(int argc, char** argv) {
	Mat frame = imread("photo/persons.png");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", frame);

	Net net = readNetFromTensorflow(model_bin, config_text);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	Mat blobImage = blobFromImage(frame, 1.0,
		Size(300, 300),
		Scalar(104.0, 177.0, 123.0), false, false);

	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
			putText(frame, format(" confidence %.2f", confidence),
				Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
		}
	}
	imshow("ssd-face-detection", frame);

	waitKey(0);
	return 0;
}