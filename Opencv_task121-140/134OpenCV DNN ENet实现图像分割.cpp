#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;
std::vector<Vec3b> colors;

void showLegend();
void colorizeSegmentation(const Mat &score, Mat &segm);
String enet_model = "D:/projects/models/enet/model-best.net";
int main(int argc, char** argv)
{
	Mat frame = imread("D:/projects/models/enet/test.png");
	Net net = readNetFromTorch(enet_model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Create a window
	static const std::string kWinName = "ENet-Demo";
	namedWindow(kWinName, WINDOW_AUTOSIZE);
	imshow("input", frame);

	// Process frames.
	Mat blob = blobFromImage(frame, 0.00392, Size(1024, 512), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	Mat score = net.forward();

	Mat segm;
	colorizeSegmentation(score, segm);

	resize(segm, segm, frame.size(), 0, 0, INTER_NEAREST);
	addWeighted(frame, 0.1, segm, 0.9, 0.0, frame);

	// Put efficiency information.
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

	imshow("ENet-Demo", frame);
	if (!classes.empty())
		showLegend();
	waitKey(0);
	return 0;
}

void colorizeSegmentation(const Mat &score, Mat &segm)
{
	const int rows = score.size[2];
	const int cols = score.size[3];
	const int chns = score.size[1];

	if (colors.empty())
	{
		// Generate colors.
		colors.push_back(Vec3b());
		for (int i = 1; i < chns; ++i)
		{
			Vec3b color;
			for (int j = 0; j < 3; ++j)
				color[j] = (colors[i - 1][j] + rand() % 256) / 2;
			colors.push_back(color);
		}
	}
	else if (chns != (int)colors.size())
	{
		CV_Error(Error::StsError, format("Number of output classes does not match "
			"number of colors (%d != %zu)", chns, colors.size()));
	}

	Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
	Mat maxVal(rows, cols, CV_32FC1, score.data);
	for (int ch = 1; ch < chns; ch++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float *ptrScore = score.ptr<float>(0, ch, row);
			uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
			float *ptrMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}

	segm.create(rows, cols, CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
		Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
		for (int col = 0; col < cols; col++)
		{
			ptrSegm[col] = colors[ptrMaxCl[col]];
		}
	}
}

void showLegend()
{
	static const int kBlockHeight = 30;
	static Mat legend;
	if (legend.empty())
	{
		const int numClasses = (int)classes.size();
		if ((int)colors.size() != numClasses)
		{
			CV_Error(Error::StsError, format("Number of output classes does not match "
				"number of labels (%zu != %zu)", colors.size(), classes.size()));
		}
		legend.create(kBlockHeight * numClasses, 200, CV_8UC3);
		for (int i = 0; i < numClasses; i++)
		{
			Mat block = legend.rowRange(i * kBlockHeight, (i + 1) * kBlockHeight);
			block.setTo(colors[i]);
			putText(block, classes[i], Point(0, kBlockHeight / 2), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255, 255, 255));
		}
		namedWindow("Legend", WINDOW_NORMAL);
		imshow("Legend", legend);
	}
}
