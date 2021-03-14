#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

void knn_test(Mat& data, Mat& labels);
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


	// 开始KNN训练
	printf("Start to knn train...\n");
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(5);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);
	knn->train(tdata);
	knn->save("D:/vcworkspaces/knn_knowledge.yml");
	printf("Finished KNN...\n");

	// 测试KNN
	printf("start to test knn...\n");
	knn_test(images, labels);

	waitKey(0);
	return true;
}

void knn_test(Mat& data, Mat& labels) {
	// 加载KNN分类器
	Ptr<ml::KNearest> knn = Algorithm::load<ml::KNearest>("photo/knn_knowledge.yml");
	Mat result;
	knn->findNearest(data, 5, result);
	float count = 0;
	for (int row = 0; row < result.rows; row++) {
		int predict = result.at<float>(row, 0);
		if (labels.at<int>(row, 0) == predict) {
			count++;
		}
	}
	printf("test acc of digit numbers : %.2f \n ", (count / result.rows));

	// real test it
	Mat t1 = imread("D:/images/knn_01.png", IMREAD_GRAYSCALE);
	Mat t2 = imread("D:/images/knn_02.png", IMREAD_GRAYSCALE);
	imshow("t1", t1);
	imshow("t2", t2);
	Mat m1, m2;
	resize(t1, m1, Size(20, 20));
	resize(t2, m2, Size(20, 20));
	Mat testdata = Mat::zeros(2, 400, CV_8UC1);
	Mat testlabels = Mat::zeros(2, 1, CV_32SC1);
	Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	Mat one = m1.reshape(1, 1);
	Mat two = m2.reshape(1, 1);
	one.copyTo(testdata(rect));
	rect.y = 1;
	two.copyTo(testdata(rect));
	testlabels.at<int>(0, 0) = 1;
	testlabels.at<int>(1, 0) = 2;
	testdata.convertTo(testdata, CV_32F);

	Mat result2;
	knn->findNearest(testdata, 5, result2);
	for (int i = 0; i < result2.rows; i++) {
		int predict = result2.at<float>(i, 0);
		printf("knn t%d predict : %d, actual label ：%d \n", (i + 1), predict, testlabels.at<int>(i, 0));
	}

}

