#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

Mat clipBorders(Mat img);
Mat preprocessImage(Mat img);
void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255));
void mergeRelatedLines(vector<Vec2f> *lines, Mat &img);
vector<int> scanPuzzle(char* filename);