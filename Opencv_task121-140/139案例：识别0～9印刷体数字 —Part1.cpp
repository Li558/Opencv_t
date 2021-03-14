#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

vector<String> labels;
vector<vector<float>> text_features;
vector<float> extractFeatureData(Mat &txtImage);
float getWeightBlackNumber(Mat &data, float width, float height, float x, float y, float xstep, float ystep);
void train_data();
void test_data();
int predictDigit(vector<float> &vf);
int main(int argc, char** argv) {
	Mat src = imread("photo/td1.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	// 训练
	train_data();

	// 测试
	test_data();

	waitKey(0);
	return 0;
}

void test_data() {
	Mat src1 = imread("photo/digit-02.png");
	imshow("input image", src1);

	Mat gray, binary;
	cvtColor(src1, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	vector<vector<Point>> contours;
	vector<Vec4i> hiearchy;
	findContours(binary.clone(), contours, hiearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (size_t t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);
		//Mat roi = binary(rect);
		//bitwise_not(roi, roi);
		vector<float> vf = extractFeatureData(binary(rect));
		int labelIndex = predictDigit(vf);
		printf("current digit is : %s\n", labels[labelIndex].c_str());
		if (labelIndex >= 0) {
			putText(src1, labels[labelIndex].c_str(), Point(rect.tl().x, rect.br().y + 15), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
		}
		else {
			putText(src1, "U", Point(rect.br().x, rect.br().y + 10), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
		}
	}
	imshow("识别结果", src1);
}

int predictDigit(vector<float> &vf) {
	float mindist = 100000000;
	int labelIndex = -1;
	for (int i = 0; i < text_features.size(); i++) {
		float dist = 0;
		vector<float> temp = text_features[i];
		for (int k = 0; k < vf.size(); k++) {
			float d = temp[k] - vf[k];
			dist += (d*d);
		}
		if (mindist > dist) {
			mindist = dist;
			labelIndex = i;
		}
	}
	return labelIndex;
}

void train_data() {
	Mat src1 = imread("D:/images/td1.png");
	Mat gray, binary;
	cvtColor(src1, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	vector<vector<Point>> contours;
	vector<Vec4i> hiearchy;
	findContours(binary.clone(), contours, hiearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	Mat result; //= Mat::zeros(src.size(), CV_8UC3);
	cvtColor(binary, result, COLOR_GRAY2BGR);
	vector<Rect> rects;
	for (size_t t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);
		rects.push_back(rect);
		rectangle(src1, rect, Scalar(0, 0, 255), 1);
	}
	// sort
	for (int i = 0; i < rects.size() - 1; i++) {
		for (int j = i + 1; j < rects.size(); j++) {
			if (rects[i].x > rects[j].x) {
				Rect tmp = rects[j];
				rects[j] = rects[i];
				rects[i] = tmp;
			}
		}
	}

	for (size_t t = 0; t < rects.size(); t++) {
		//Mat roi = binary(rects[t]);
		//bitwise_not(roi, roi);
		vector<float> vf = extractFeatureData(binary(rects[t]));
		for (int i = 0; i < vf.size(); i++) {
			printf("Num. %d, vf = \n", t);
			printf("vf[%d]=%.4f\n", i, vf[i]);
		}
		printf("\n");
		text_features.push_back(vf);
		if (t == (rects.size() - 1))
			labels.push_back("0");
		else
			labels.push_back(format("%d", (t + 1)));
	}
	imshow("train data", src1);
}

vector<float> extractFeatureData(Mat &txtImage) {
	// total black pixels;
	vector<float> vectorData(40);
	float width = txtImage.cols;
	float height = txtImage.rows;

	// vector data
	float bins = 10.0f;
	float xstep = width / 4.0f;
	float ystep = height / 5.0f;

	int index = 0;
	for (float y = 0; y < height; y += ystep) {
		for (float x = 0; x < width; x += xstep) {
			vectorData[index] = getWeightBlackNumber(txtImage, width, height, x, y, xstep, ystep);
			index++;
		}
	}

	// calculate Y Project
	xstep = width / bins;
	for (float x = 0; x < width; x += xstep) {
		if ((x + xstep) - width > 1) continue;
		vectorData[index] = getWeightBlackNumber(txtImage, width, height, x, 0, xstep, height);
		index++;
	}

	// calculate X Project
	ystep = height / bins;
	for (float y = 0; y < height; y += ystep) {
		if ((y + ystep) - height > 1) continue;
		vectorData[index] = getWeightBlackNumber(txtImage, width, height, 0, y, width, ystep);
		index++;
	}

	// normalization vector data
	float sum = 0;
	// 4x5 cell = 20 vector
	for (int i = 0; i < 20; i++) {
		sum += vectorData[i];
	}
	for (int i = 0; i < 20; i++) {
		vectorData[i] = vectorData[i] / sum;
	}

	// Y Projection 10 vector
	sum = 0;
	for (int i = 20; i < 30; i++) {
		sum += vectorData[i];
	}
	for (int i = 20; i < 30; i++) {
		vectorData[i] = vectorData[i] / sum;
	}

	// X Projection 10 vector
	sum = 0;
	for (int i = 30; i < 40; i++) {
		sum += vectorData[i];
	}
	for (int i = 30; i < 40; i++) {
		vectorData[i] = vectorData[i] / sum;
	}
	return vectorData;
}

float getWeightBlackNumber(Mat &data, float width, float height, float x, float y, float xstep, float ystep) {
	float weightNum = 0;

	// 取整
	int nx = (int)floor(x);
	int ny = (int)floor(y);

	// 浮点数
	float fx = x - nx;
	float fy = y - ny;

	// 计算位置
	float w = x + xstep;
	float h = y + ystep;
	if (w > width) {
		w = width - 1;
	}
	if (h > height) {
		h = height - 1;
	}

	// 权重取整
	int nw = (int)floor(w);
	int nh = (int)floor(h);

	// 浮点数
	float fw = w - nw;
	float fh = h - nh;

	// 计算
	int c = 0;
	int ww = (int)width;
	float weight = 0;
	int row = 0;
	int col = 0;
	for (row = ny; row < nh; row++) {
		for (col = nx; col < nw; col++) {
			c = data.at<uchar>(row, col);
			if (c == 0) {
				weight++;
			}
		}
	}

	float w1 = 0, w2 = 0, w3 = 0, w4 = 0;
	// calculate w1
	if (fx > 0) {
		col = nx + 1;
		if (col > width - 1) {
			col = col - 1;
		}
		float count = 0;
		for (row = ny; row < nh; row++) {
			c = data.at<uchar>(row, col);
			if (c == 0) {
				count++;
			}
		}
		w1 = count * fx;
	}

	// calculate w2
	if (fy > 0) {
		row = ny + 1;
		if (row > height - 1) {
			row = row - 1;
		}
		float count = 0;
		for (col = nx; col < nw; col++) {
			c = data.at<uchar>(row, col);
			if (c == 0) {
				count++;
			}
		}
		w2 = count * fy;
	}

	// calculate w3
	if (fw > 0) {
		col = nw + 1;
		if (col > width - 1) {
			col = col - 1;
		}
		float count = 0;
		for (row = ny; row < nh; row++) {
			c = data.at<uchar>(row, col);
			if (c == 0) {
				count++;
			}
		}
		w3 = count * fw;
	}

	// calculate w4
	if (fh > 0) {
		row = nh + 1;
		if (row > height - 1) {
			row = row - 1;
		}
		float count = 0;
		for (col = nx; col < nw; col++) {
			c = data.at<uchar>(row, col);
			if (c == 0) {
				count++;
			}
		}
		w4 = count * fh;
	}
	weightNum = (weight - w1 - w2 + w3 + w4);
	if (weightNum < 0) {
		weightNum = 0;
	}
	return weightNum;
}