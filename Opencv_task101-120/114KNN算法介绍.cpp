#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc, char** argv) {
	Mat data = imread("photo/digits.png");
	Mat gray;
	cvtColor(data, gray, COLOR_BGR2GRAY);

	// 分割为5000个cells
	Mat images = Mat::zeros(5000, 400, CV_8UC1);
	Mat labels = Mat::zeros(5000, 1, CV_8UC1);
	Rect rect;
	rect.height = 20;
	rect.width = 20;
	int index = 0;
	Rect roi;
	roi.x = 0;
	roi.height = 1;
	roi.width = 400;
	for (int row = 0; row < 50; row++) {
		int label = row / 5;
		for (int col = 0; col < 100; col++) {
			Mat digit = Mat::zeros(20, 20, CV_8UC1);
			index = row * 100 + col;
			rect.x = col * 20;
			rect.y = row * 20;
			gray(rect).copyTo(digit);
			Mat one_row = digit.reshape(1, 1);
			roi.y = index;
			one_row.copyTo(images(roi));
			labels.at<uchar>(index, 0) = label;
		}
	}
	printf("load sample hand-writing data...\n");

	// 转换为浮点数
	images.convertTo(images, CV_32FC1);
	labels.convertTo(labels, CV_32SC1);

	// 开始KNN训练
	printf("Start to knn train...\n");
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(5);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);
	knn->train(tdata);
	knn->save("model/knn_knowledge.yml");
	printf("Finished KNN...\n");
	return true;
}