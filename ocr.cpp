#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <math.h>
#include <conio.h>
#include <iostream>
#include <array>

using namespace cv;
using namespace std;


// ------------------- TESTS -------------------
void testPixels(Mat, Mat); // TODO: erase
void imageDimensions(const Mat&); // TODO: erase

// ------------------- UTIL -------------------
void displayImage(String, const Mat&);
Mat getBinaryImage(const Mat&);
inline bool fileExists(const std::string&);

// ------------------- PREPROCESSING -------------------
Mat preProcessing(const Mat&, int, int);
Mat preProcessing(const Mat&);
Rect findBoundingBox(const Mat&);
void findWords(const Mat&);
void findX(const Mat&, int*, int*);
void findY(const Mat&, int*, int*);
Mat invert(const Mat&);
void labelPixel(const Mat&, Mat&, Mat&, int, int, int&);
int labelPixels(const Mat&, Mat&);
void relabelPixels(const Mat&, Mat&, int);
Mat rescale(const Mat&, int, int);

// ------------------- simple OCR -------------------
class SimpleOCR{
public:
	static const int dimDigits = 40, dimLetters = 20;
	
	SimpleOCR();

	float classify(const Mat& img, bool showResult, bool isDigit, bool isPreProcessed);
	char classifyWord(const Mat& srcImage);

private:
	char folderPath[255], correspondingCharacters[62];
	int digitsTrainSamples, digitsClasses;
	static const int K = 10;
	KNearest *knnDigits, *knnLetters;
	Mat digitsTrainData, digitsTrainLabels, lettersTrainData, lettersTrainLabels;

	void getDigitsData();
	void getLettersData();
	void testDigits();
	void testLetters();
	void train();
};


// ----------------------- global OCR -----------------------
SimpleOCR ocr;

// ----------------------- MAIN -----------------------
int main()
{
	Mat im = imread("../images/tests/modal.png");
	findWords(im);

	cout << "\n\n ---------------------------------------------------------------\n"
	"|\tClass\t|\tPrecision\t|\tAccuracy\t|\n"
	" ---------------------------------------------------------------\n";
	im = imread("../images/tests/1.jpg");
	ocr.classify(im, true, true, false);
	im = imread("../images/tests/3.jpg");
	ocr.classify(im, true, true, false);
	im = imread("../images/tests/6.jpg");
	ocr.classify(im, true, true, false);
	im = imread("../images/tests/7.jpg");
	ocr.classify(im, true, true, false);
	im = imread("../images/tests/A.jpg");
	ocr.classify(im, true, false, false);
	im = imread("../images/tests/e.jpg");
	ocr.classify(im, true, false, false);
	im = imread("../images/tests/f.jpg");
	ocr.classify(im, true, false, false);

	while (true){
		cout << "\nDigit [1], letter [2] or word [3]? ";
		char choice;
		cin >> choice;
		while (choice != '1' && choice != '2' && choice != '3'){
			cout << "Invalid choice! Digit [1], letter [2] or word [3]? ";
			cin >> choice;
		}

		bool isDigit = (choice == '1'), isWord = (choice == '3');

		char folderPath[] = "../images/tests/", userEntry[50], fullPath[200];
		cout << "Enter the name of the file: ";
		cin >> userEntry;

		sprintf(fullPath, "%s%s", folderPath, userEntry);
		if (fileExists(fullPath)){
			Mat im = imread(fullPath);
			displayImage("image", im); waitKey();
			if (isWord)
				findWords(im);
			else{
				cout << "\n\n ---------------------------------------------------------------\n"
					"|\tClass\t|\tPrecision\t|\tAccuracy\t|\n"
					" ---------------------------------------------------------------\n";
				ocr.classify(im, true, isDigit, false);
			}
		}
		else
			cout << fullPath << " does not exist!\n";
	}

	cout << "\n\nPress any key to exit...";
	_getch();
	return 0;
}


// ------------------- TESTS -------------------
void findWords(const Mat& imgSrc){
	Mat imgBin = preProcessing(imgSrc);
	displayImage("", imgBin);
	waitKey();

	Mat labels(imgBin.rows, imgBin.cols, CV_16SC1);
	labels.setTo(0);
	int numLabels = labelPixels(imgBin, labels);

	/*for (int i = 0; i < imgBin.rows; i++){
		for (int j = 0; j < imgBin.cols; j++){
			cout << (imgBin.at<bool>(i, j) == 255 ? "0" : "1");
		}
		cout << endl;
	}

	cout << "labels:\n";
	for (int i = 0; i < imgBin.rows; i++){
		for (int j = 0; j < imgBin.cols; j++){
			cout << labels.at<short>(i, j);
		}
		cout << endl;
	}*/
	

	for (int i = 0; i < imgBin.cols; i++)
		relabelPixels(imgBin, labels, i);

	cout << "numLabels = " << numLabels << endl;
	int *xMin = new int[numLabels], *xMax = new int[numLabels], *yMin = new int[numLabels], *yMax = new int[numLabels];
	for (int i = 0; i < numLabels; i++){
		xMin[i] = imgBin.cols;
		yMin[i] = imgBin.rows;
		xMax[i] = 0;
		yMax[i] = 0;
	}


	for (int i = 0; i < imgBin.rows; i++){
		for (int j = 0; j < imgBin.cols; j++){
			int l = labels.at<short>(i, j);
			if (l != 0){
				l--;
				if (xMin[l] > j)
					xMin[l] = j;
				else if (xMax[l] < j)
					xMax[l] = j;

				if (yMin[l] > i)
					yMin[l] = i;
				else if (yMax[l] < i)
					yMax[l] = i;

			}
		}
	}

	/*
	for (int i = 0; i < numLabels; i++){
		int x = xMin[i], y = yMin[i],
			x2 = xMax[i], y2 = yMax[i];
		cout << x << " * " << y << " * " << x2 << " * " << y2 << endl;
	}
	*/

	cout << "\n\n ---------------------------------------------------------------\n"
		"|\tClass\t|\tPrecision\t|\tAccuracy\t|\n"
		" ---------------------------------------------------------------\n";

	char *word = new char[numLabels+1];
	for (int i = 0; i < numLabels; i++){
		int x = xMin[i], y = yMin[i],
			w = xMax[i] - x + 1, h = yMax[i] - y + 1;
		Mat imgLetter = rescale(imgBin(Rect(x, y, w, h)), SimpleOCR::dimLetters, SimpleOCR::dimLetters);
		char res = ocr.classifyWord(imgLetter);
		word[i] = res;
	}word[numLabels] = '\0';
	
	cout << "OCR result = " << word << endl;

	delete[] word, xMin, xMax, yMin, yMax;
}

void testPixels(Mat I, Mat Ibw){
	cout << "\n\n[testing pixel print]\n";
	int y = 60, x = 30;
	Vec3b p1 = I.at<Vec3b>(y, x),
		p2 = Ibw.at<Vec3b>(y, x);

	cout << y << " * " << x << "\n";
	cout << p1 << p2;
}

void imageDimensions(const Mat& I){
	cout << "width (cols) = " << I.cols << ", height (rows) = " << I.rows << endl;
}


// ------------------- UTIL -------------------
void displayImage(String name, const Mat& I){
	namedWindow(name);
	imshow(name, I);
}

Mat getBinaryImage(const Mat& imgSrc){
	Mat	imgGrayScale;
	cvtColor(imgSrc, imgGrayScale, CV_BGR2GRAY); // convert BGR to gsray

	Mat imgBinary(imgGrayScale.size(), imgGrayScale.type()); // binary image
	threshold(imgGrayScale, imgBinary, 100, 255, cv::THRESH_BINARY); // apply threshold

	return imgBinary;
}

inline bool fileExists(const std::string& name) {
	if (FILE *file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}else return false;
}


// ------------------- PREPROCESSING -------------------

// bounding box + rescale
Mat preProcessing(const Mat& imgSrc, int finalWidth, int finalHeight){
	Mat imgBin = getBinaryImage(imgSrc);

	Rect box = findBoundingBox(imgBin);
	Mat subRect(imgBin, Rect(box.x, box.y, box.width, box.height));

	int dim = (box.width > box.height ? box.width : box.height);
	Mat result = cvCreateImage(cvSize(dim, dim), 8, 1); // last parameter = 1 for binary, 3 for RGB!
	result = Scalar(255, 255, 255);

	int x = (int)floor((float)(dim - box.width) / 2.),
		y = (int)floor((float)(dim - box.height) / 2.);

	subRect.copyTo(result(Rect(x, y, subRect.cols, subRect.rows)));

	Mat scaledResult;
	resize(result, scaledResult, Size(finalWidth, finalHeight));

	// TODO: erase
	/*	displayImage("orig", imgSrc);
		displayImage("bin", imgBin);
		displayImage("subrect", subRect);
		displayImage("res", result);
		displayImage("sc res", scaledResult);
		waitKey();
		*/
	return scaledResult;
}

// bounding box, no rescale
Mat preProcessing(const Mat& imgSrc){
	Mat imgBin = getBinaryImage(imgSrc);

	Rect box = findBoundingBox(imgBin);
	Mat subRect(imgBin, Rect(box.x, box.y, box.width, box.height));

	return subRect;
}

Rect findBoundingBox(const Mat& imgSrc){
	int xmin, xmax, ymin, ymax;
	xmin = xmax = ymin = ymax = 0;

	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);

	Rect rect = Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	return rect;
}

void findX(const Mat& imgSrc, int* min, int* max){
	bool minFound = 0;
	int maxVal = imgSrc.rows * 255;

	for (int i = 0; i < imgSrc.cols; i++){
		Mat col = imgSrc.col(i);
		Scalar val = sum(col);
		if (val.val[0] < maxVal){
			*max = i;
			if (!minFound){
				minFound = true;
				*min = i;
			}
		}
	}
}

void findY(const Mat& imgSrc, int* min, int* max){
	bool minFound = 0;
	int maxVal = imgSrc.cols * 255;

	for (int i = 0; i < imgSrc.rows; i++){
		Mat row = imgSrc.row(i);
		Scalar val = sum(row);
		if (val.val[0] < maxVal){
			*max = i;
			if (!minFound){
				minFound = true;
				*min = i;
			}
		}
	}
}

Mat invert(const Mat& img){
	return Scalar::all(255) - img;
}

void labelPixel(const Mat& img, Mat& labels, Mat& passed, int i, int j, int& pixelLabel){
	passed.at<bool>(i, j) = 1;
	int l = labels.at<short>(i, j);
	if (l == 0){
		labels.at<short>(i, j) = pixelLabel;
		l = pixelLabel;
		pixelLabel++;
	}

	int up = (i>0 ? i - 1 : i), down = (i < img.rows - 1 ? i + 1 : i),
		left = (j>0 ? j - 1 : j), right = (j < img.cols - 1 ? j + 1 : j);

	//cout << i << " * " << j << " udlr = " << up << " " << down << " " << left << " " << right << endl;
	for (int ii = up; ii <= down; ii++){
		for (int jj = left; jj <= right; jj++){
			if (img.at<bool>(ii, jj) != 255 && !passed.at<bool>(ii, jj)){
				labels.at<short>(ii, jj) = l;
				labelPixel(img, labels, passed, ii, jj, pixelLabel);
			}
		}
	}
}

int labelPixels(const Mat& img, Mat& labels){
	int pixelLabel = 1;
	Mat passed(img.rows, img.cols, CV_8U);
	passed.setTo(0);
	for (int j = 0; j < img.cols; j++){
		for (int i = 0; i < img.rows; i++){
			if (img.at<bool>(i, j) != 255 && labels.at<short>(i, j) == 0 && !passed.at<bool>(i,j))
				labelPixel(img, labels, passed, i, j, pixelLabel);
		}
	}

	return pixelLabel-1;
}

void relabelPixels(const Mat& img, Mat& labels, int col){

}

Mat rescale(const Mat& subRect, int finalWidth, int finalHeight){
	int dim = (subRect.rows > subRect.cols ? subRect.rows: subRect.cols);
	Mat result = cvCreateImage(cvSize(dim, dim), 8, 1); // last parameter = 1 for binary, 3 for RGB!
	result = Scalar(255, 255, 255);

	int x = (int)floor((float)(dim - subRect.cols) / 2.),
		y = (int)floor((float)(dim - subRect.rows) / 2.);

	subRect.copyTo(result(Rect(x, y, subRect.cols, subRect.rows)));

	Mat scaledResult;
	resize(result, scaledResult, Size(finalWidth, finalHeight));

	return scaledResult;
}
// ------------------- simple OCR -------------------

// constructor
SimpleOCR::SimpleOCR(){
	sprintf(folderPath, "../images/samples");
	digitsTrainSamples = 50;
	digitsClasses = 10;

	for (int i = 0; i < 62; i++){
		if (i < 10)
			correspondingCharacters[i] = '0' + i;
		else if (i < 35)
			correspondingCharacters[i] = 'A' + i - 10;
		else
			correspondingCharacters[i] = 'a' + i - 36;
	}

	getDigitsData();
	getLettersData();
	train();
	testDigits();
	testLetters();
}

// public
char SimpleOCR::classifyWord(const Mat& srcImage){
	float result = classify(srcImage, true, false, true);
	char resultChar = correspondingCharacters[(int)result];
	return resultChar;
}

float SimpleOCR::classify(const Mat& srcImage, bool showResult, bool isDigit, bool isPreProcessed){
	Mat procImage, floatData, nearest(1, K, CV_32FC1);

	int dim = (isDigit ? dimDigits : dimLetters);
	procImage = (isPreProcessed ? srcImage : preProcessing(srcImage, dim, dim));
	procImage.convertTo(floatData, CV_32FC1);

	float result = (isDigit ? knnDigits : knnLetters)->find_nearest(floatData.reshape(1, 1), K, 0, 0, &nearest, 0);

	int accuracy = 0;
	for (int i = 0; i < nearest.rows; i++)
		for (int j = 0; j < nearest.cols; j++)
			if (nearest.at<float>(i, j) == result)
				accuracy++;

	float precision = 100 * ((float)accuracy / (float)K);
	char resultChar = correspondingCharacters[(int) result];
	if (showResult){
		printf("|\t%c\t| \t%.2f%%  \t| \t%d of %d \t| \n", resultChar, precision, accuracy, K);
		printf(" ---------------------------------------------------------------\n");
	}

	return result;
}

// private
void SimpleOCR::getDigitsData(){
	cout << "Fetching digits data...\n";

	for (int i = 0; i < digitsClasses; i++){
		for (int j = 0; j < digitsTrainSamples; j++){
			Mat srcImage, procImage;
			char imgPath[255];
			if (j < 10)
				sprintf(imgPath, "%s/digits/%d/%d0%d.pbm", folderPath, i, i, j);
			else
				sprintf(imgPath, "%s/digits/%d/%d%d.pbm", folderPath, i, i, j);
			srcImage = imread(imgPath);
			srcImage = imread(imgPath);
			procImage = preProcessing(srcImage, dimDigits, dimDigits);

			Mat floatData;
			procImage.convertTo(floatData, CV_32FC1); // convert to float
			digitsTrainData.push_back(floatData.reshape(1, 1)); // add a row to train data (flattened image)
			digitsTrainLabels.push_back(i); // add label (class): 0 to 9
		}
	}
}

void SimpleOCR::getLettersData(){
	cout << "Fetching letters data...\n";

	char uppercaseLetters[] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
	int nLowercaseSamples[] = { 200, 200, 200, 200, 201, 200, 200, 200,   7,  11, 200, 200, 202, 200, 200, 198, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200 },
		nUppercaseSamples[] = { 200, 200, 198, 200, 198, 201, 195, 200, 201, 200, 200, 200, 200, 199, 200, 200, 200, 200, 200, 199, 200, 200, 200, 200, 200, 199 };

	for (int i = 0; i < 26; i++){
		char letter = uppercaseLetters[i];
		int n = nUppercaseSamples[i] / 2;
		for (int j = 0; j < n; j++){
			Mat srcImage, procImage;
			char imgPath[255];
			sprintf(imgPath, "%s/upper/%c/%d.png", folderPath, letter, j);

			srcImage = imread(imgPath);
			procImage = preProcessing(srcImage, dimLetters, dimLetters);
			procImage = invert(procImage);

			Mat floatData;
			procImage.convertTo(floatData, CV_32FC1); // convert to float
			lettersTrainData.push_back(floatData.reshape(1, 1)); // add a row to train data (flattened image)
			lettersTrainLabels.push_back(10 + i); // add label (class): A (10) to Z (35)
		}
	}

	for (int i = 0; i < 26; i++){
		char letter = uppercaseLetters[i];
		int n = nLowercaseSamples[i] / 2;
		for (int j = 0; j < n; j++){
			Mat srcImage, procImage;
			char imgPath[255];
			sprintf(imgPath, "%s/lower/%c/%d.png", folderPath, letter, j);

			srcImage = imread(imgPath);
			procImage = preProcessing(srcImage, dimLetters, dimLetters);
			procImage = invert(procImage);

			Mat floatData;
			procImage.convertTo(floatData, CV_32FC1); // convert to float
			lettersTrainData.push_back(floatData.reshape(1, 1)); // add a row to train data (flattened image)
			lettersTrainLabels.push_back(36 + i); // add label (class): a (36) to z (61)
		}
	}
}

void SimpleOCR::testDigits(){
	cout << "Testing handwritten digits results...\n";

	int error = 0, testCount = 0;
	for (int i = 0; i < digitsClasses; i++){
		for (int j = 50; j < 50 + digitsTrainSamples; j++){
			Mat srcImage, procImage;
			char imgPath[255];
			sprintf(imgPath, "%s/digits/%d/%d%d.pbm", folderPath, i, i, j);
			srcImage = imread(imgPath);

			float r = classify(srcImage, false, true, false);
			if ((int)r != i)
				error++;

			testCount++;
		}
	}

	float totalError = 100 * (float)error / (float)testCount;
	cout << "   --> System error = " << ceil(totalError * 100) / 100. << "%\n";
}

void SimpleOCR::testLetters(){
	cout << "Testing letters results...\n";

	int error = 0, testCount = 0;

	char uppercaseLetters[] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
	int nLowercaseSamples[] = { 200, 200, 200, 200, 201, 200, 200, 200, 7, 11, 200, 200, 202, 200, 200, 198, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200 },
		nUppercaseSamples[] = { 200, 200, 198, 200, 198, 201, 195, 200, 201, 200, 200, 200, 200, 199, 200, 200, 200, 200, 200, 199, 200, 200, 200, 200, 200, 199 };

	for (int i = 0; i < 26; i++){
		char letter = uppercaseLetters[i];
		int n = nUppercaseSamples[i];
		for (int j = n / 2; j < min(n, n/2+50); j++){
			Mat srcImage, procImage;
			char imgPath[255];
			sprintf(imgPath, "%s/upper/%c/%d.png", folderPath, letter, j);

			srcImage = imread(imgPath);
			srcImage = invert(srcImage);

			float r = classify(srcImage, false, false, false);
			//cout << "\tValue = " << i << "\tKNN result = " << r << endl;
			if ((int)r != i + 10)
				error++;

			testCount++;
		}
	}

	for (int i = 0; i < 26; i++){
		char letter = uppercaseLetters[i];
		int n = nLowercaseSamples[i];
		for (int j = n / 2; j < min(n, n / 2 + 50); j++){
			Mat srcImage, procImage;
			char imgPath[255];
			sprintf(imgPath, "%s/lower/%c/%d.png", folderPath, letter, j);

			srcImage = imread(imgPath);
			srcImage = invert(srcImage);

			float r = classify(srcImage, false, false, false);
			//cout << "\tValue = " << i << "\tKNN result = " << r << endl;
			if ((int)r != i + 36)
				error++;

			testCount++;
		}
	}

	float totalError = 100 * (float)error / (float)testCount;
	cout << "   --> System error = " << ceil(totalError * 100) / 100. << "%\n";
}

void SimpleOCR::train(){
	cout << "Training data...\n";
	knnDigits = new KNearest(digitsTrainData, digitsTrainLabels);
	knnLetters = new KNearest(lettersTrainData, lettersTrainLabels);
}
