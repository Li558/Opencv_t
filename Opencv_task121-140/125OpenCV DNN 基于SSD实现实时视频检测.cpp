#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;
String labelFile = "models/ssd/labelmap_det.txt";
String modelFile = "models/ssd/MobileNetSSD_deploy.caffemodel";
String model_text_file = "models/ssd/MobileNetSSD_deploy.prototxt";

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) {
	// load model
	Net net = readNetFromCaffe(model_text_file, modelFile);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	VideoCapture cap = VideoCapture(0);
	Mat frame;
	while (true) {
		bool ret = cap.read(frame);
		if (!ret) break;
		Mat blobImage = blobFromImage(frame, 0.007843,
			Size(300, 300),
			Scalar(127.5, 127.5, 127.5), true, false);
		printf("blobImage width : %d, height: %d\n", blobImage.size[2], blobImage.size[3]);

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
				putText(frame, format(" confidence %.2f, %s", confidence, objNames[objIndex].c_str()), Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, 8);
			}
		}
		imshow("ssd-video-demo", frame);
		char c = waitKey(10);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}