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

	int index = 0;
	Rect roi;
	roi.x = 0;
	roi.height = 1;
	roi.width = 400;
	for (int row = 0; row < 50; row++) {
		int label = row / 5;
		int offsety = row * 20;
		for (int col = 0; col < 100; col++) {
			int offsetx = col * 20;
			Mat digit = Mat::zeros(Size(20, 20), CV_8UC1);
			for (int sr = 0; sr < 20; sr++) {
				for (int sc = 0; sc < 20; sc++) {
					digit.at<uchar>(sr, sc) = gray.at<uchar>(sr + offsety, sc + offsetx);
				}
			}
			Mat one_row = digit.reshape(1, 1);
			printf("index : %d \n", index);
			roi.y = index;
			one_row.copyTo(images(roi));
			labels.at<uchar>(index, 0) = label;
			index++;
		}
	}
	printf("load sample hand-writing data...\n");
	//imwrite("D:/result.png", images);

	// 转换为浮点数
	images.convertTo(images, CV_32FC1);
	labels.convertTo(labels, CV_32SC1);

	printf("load sample hand-writing data...\n");


	// 开始训练
	printf("Start to Random Trees train...\n");
	Ptr<RTrees> model = RTrees::create();
	/*model->setMaxDepth(10);
	model->setMinSampleCount(10);
	model->setRegressionAccuracy(0);
	model->setUseSurrogates(false);
	model->setMaxCategories(15);
	model->setPriors(Mat());
	model->setCalculateVarImportance(true);
	model->setActiveVarCount(4);
	*/
	TermCriteria tc = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.01);
	model->setTermCriteria(tc);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);
	model->train(tdata);
	model->save("model/rtrees_knowledge.yml");
	printf("Finished Random trees...\n");

	waitKey(0);
	return true;
}