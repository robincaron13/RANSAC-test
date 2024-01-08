#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <random>
#include <cstdint>

#include "GRANSAC.h"
#include "LineModel.h"

GRANSAC::VPFloat Slope(int x0, int y0, int x1, int y1)
{
	return (GRANSAC::VPFloat)(y1 - y0) / (x1 - x0);
}

void DrawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int LineWidth)
{
	GRANSAC::VPFloat slope = Slope(a.x, a.y, b.x, b.y);

	cv::Point p(0, 0), q(img.cols, img.rows);

	p.y = -(a.x - p.x) * slope + a.y;
	q.y = -(b.x - q.x) * slope + b.y;

	cv::line(img, p, q, color, LineWidth, cv::LINE_AA, 0);
}

int main(int argc, char* argv[])
{
	if (argc != 1 && argc != 3)
	{
		std::cout << "[ USAGE ]: " << argv[0] << " [<Image Size> = 1000] [<nStars> = 500]" << std::endl;
		return -1;
	}

	int Side = 600;
	int nStars = 6;
	int nMegaConstStars = 20;

	if (argc == 3)
	{
		Side = std::atoi(argv[1]);
		nStars = std::atoi(argv[2]);
	}

	int radiusPoints = floor(Side / 200) ;
	int radiusPointsBest = floor(Side / 200) + 1;


	cv::Mat Canvas(Side, Side, CV_8UC3);
	Canvas.setTo(255);

	// Randomly generate points in a 2D plane roughly aligned in a line for testing
	std::random_device SeedDevice;
	std::mt19937 RNG = std::mt19937(SeedDevice());

	std::uniform_int_distribution<int> DataDist(100, Side - 100); // [Incl, Incl]
	//std::binomial_distribution<int> DataDist( Side - 1, 0.5);

	int Perturb = 2;
	std::normal_distribution<GRANSAC::VPFloat> PerturbDist(0, Perturb);

	int MegaConstPerturb = 400;
	std::uniform_int_distribution<int> PerturbDistMegaConst(0, MegaConstPerturb); // [Incl, Incl]

	std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
	for (int i = 0; i < nStars; ++i)
	{
		int Diag = DataDist(RNG);
		cv::Point Pt(floor(Diag + PerturbDist(RNG)), floor(Diag + PerturbDist(RNG)));
		cv::circle(Canvas, Pt, radiusPoints, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

		std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(Pt.x, Pt.y);
		CandPoints.push_back(CandPt);
	}

	for (int k = 0; k < nMegaConstStars; ++k)
	{
		int Diag = DataDist(RNG);
		cv::Point Pt(floor(Diag + PerturbDistMegaConst(RNG)), floor(Diag + PerturbDistMegaConst(RNG)));
		cv::circle(Canvas, Pt, radiusPoints, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

		std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(Pt.x, Pt.y);
		CandPoints.push_back(CandPt);
	}

	int Threshold = 5;
	int Iterations = 100;
	GRANSAC::RANSAC<Line2DModel, 2> Estimator;
	Estimator.Initialize(Threshold, Iterations); // Threshold, iterations
	int64_t start = cv::getTickCount();
	Estimator.Estimate(CandPoints);
	int64_t end = cv::getTickCount();
	std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms, with (Threshold, Iterations) = (" << Threshold <<", " << Iterations << ")" << std::endl;

	auto BestInliers = Estimator.GetBestInliers();
	if (BestInliers.size() > 0)
	{
		for (auto& Inlier : BestInliers)
		{
			auto RPt = std::dynamic_pointer_cast<Point2D>(Inlier);
			cv::Point Pt(floor(RPt->m_Point2D[0]), floor(RPt->m_Point2D[1]));
			cv::circle(Canvas, Pt, radiusPointsBest, cv::Scalar(120, 225, 103), -1, cv::LINE_AA);
		}
	}

	auto BestLine = Estimator.GetBestModel();
	if (BestLine)
	{
		auto BestLinePt1 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[0]);
		auto BestLinePt2 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[1]);
		if (BestLinePt1 && BestLinePt2)
		{
			cv::Point Pt1(BestLinePt1->m_Point2D[0], BestLinePt1->m_Point2D[1]);
			cv::Point Pt2(BestLinePt2->m_Point2D[0], BestLinePt2->m_Point2D[1]);
			DrawFullLine(Canvas, Pt1, Pt2, cv::Scalar(0, 0, 255), 2);
		}
	}

	cv::imwrite("LineFitting_test.png", Canvas);

	/*while (true)
	{
		cv::imshow("RANSAC Example", Canvas);

		char Key = cv::waitKey(1);
		if (Key == 27)
			return 0;
		if (Key == ' ')
			cv::imwrite("LineFitting.png", Canvas);
	}*/

	return 0;
}

