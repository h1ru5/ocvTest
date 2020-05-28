#include "Detection/YAED.h"
#include <conio.h>
#include <mutex>
#include <atomic>

using namespace std::chrono_literals;

std::atomic<bool> g_ImagesRead = false;
std::mutex g_ImagesMutex;

struct CImage
{
	cv::String	path;
	cv::Mat3b	mat;
};

void Read(const char* folder, std::vector<CImage>& output)
{
	cv::String strFolder(folder);

	char lastChar = strFolder.back();
	if (lastChar != '\\' && lastChar != '/')
		strFolder += '\\';

	std::vector<cv::String> files;
	cv::glob(strFolder + "*.jpg", files);

	output.reserve(files.size());
	for (size_t i = 0; i < files.size(); ++i)
	{
		g_ImagesMutex.lock();
		output.push_back({ files[i], cv::imread(files[i]) });
		g_ImagesMutex.unlock();
	}
}

bool IsGood(std::vector<cv::Rect>& bounds, const cv::Rect& r, const cv::Size& imgsize)
{
	// too small
	int m1 = std::max(imgsize.width, imgsize.height);
	int m2 = std::max(r.width, r.height);
	if ((m1 / 15) > m2)
		return false;

	// overlaps too much
	for (size_t i = 0; i < bounds.size(); ++i)
	{
		if (bounds[i].area() < ((bounds[i] & r).area() * 1.25f))
			return false;
	}
	return true;
}

void EllipsesBounds(std::vector<Ellipse>& ellipses, std::vector<cv::Rect>& bounds, const cv::Size& imgsize)
{
	for (size_t i = 0; i < ellipses.size(); ++i)
	{
		Ellipse& e = ellipses[i];
		cv::Rect r = cv::RotatedRect(cv::Point2f(e._xc, e._yc), cv::Size2f(e._a, e._b), e._rad * 180.f / (float)CV_PI).boundingRect();
		if (IsGood(bounds, r, imgsize))
			bounds.push_back(r);
	}
}

void Write(cv::String path, std::vector<cv::Rect>& bounds)
{
	cv::String noExt(path);
	size_t extI = noExt.find_last_of('.') + 1;
	noExt.erase(extI);
	noExt.append("txt");

	std::ofstream file(noExt, std::ofstream::out | std::ofstream::trunc);
	if (!file.good())
		return;

	for (size_t i = 0; i < bounds.size(); ++i)
	{
		cv::Point tl(bounds[i].x, bounds[i].y);
		cv::Point tr((bounds[i].x + bounds[i].width), bounds[i].y);
		cv::Point br((bounds[i].x + bounds[i].width), (bounds[i].y + bounds[i].height));
		cv::Point bl(bounds[i].x, (bounds[i].y + bounds[i].height));

		file << tl << ' ' << tr << ' ' << br << ' ' << bl << std::endl;
	}

	file.close();
}

void Detect(std::vector<CImage>& images)
{
	while (true)
	{
		static std::atomic<int> nextToDetect = 0;

		int index = nextToDetect++;

		bool finish = false;

		g_ImagesMutex.lock();
		bool bigger = images.size() < ((size_t)index + 1);
		g_ImagesMutex.unlock();

		while (!finish && bigger)
		{
			if (g_ImagesRead)
				finish = true;
			else
			{
				g_ImagesMutex.lock();
				bigger = images.size() < ((size_t)index + 1);
				g_ImagesMutex.unlock();

				std::this_thread::sleep_for(1ms);
			}
		}

		if (finish)
			break;


		cv::Mat1b gray;
		cvtColor(images[index].mat, gray, CV_BGR2GRAY);

		cv::Size size = images[index].mat.size();
		float taoCenters = 0.05f;
		float maxCenterDistance = sqrtf(size.width * size.width + size.height * size.height) * taoCenters;

		CYAED yaed;
		// Parameters Settings (Sect. 4.2)
		yaed.SetParameters(cv::Size(5, 5),
			1.0,
			1.0f,
			maxCenterDistance,
			16,
			3.0f,
			0.1f, // Sect. 3.3.1 - Validation
			0.62f,
			0.6f,
			16
		);

		std::vector<Ellipse> ellipses;
		yaed.Detect(gray, ellipses);

		//yaed.DrawDetectedEllipses(images[index].mat, ellipses);
		//imshow("Yaed", images[index].mat);
		//waitKey();

		std::vector<cv::Rect> bounds;
		EllipsesBounds(ellipses, bounds, images[index].mat.size());

		Write(images[index].path, bounds);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Invalid arguments. Format: ocvTest *number of threads* *path to folder 1* *path to folder N*\n";
		_getch();
		return 1;
	}

	char** folders = &argv[2];
	int readThreadCount = argc - 2;
	int detectThreadCount = atoi(argv[1]);

	std::vector<CImage> images;

	std::vector<std::thread> readThreads;
	std::vector<std::thread> detectThreads;

	for (int i = 0; i < readThreadCount; ++i)
		readThreads.push_back(std::thread(Read, folders[i], std::ref(images)));

	for (int i = 0; i < detectThreadCount; ++i)
		detectThreads.push_back(std::thread(Detect, std::ref(images)));


	for (size_t i = 0; i < readThreads.size(); ++i)
		readThreads[i].join();

	g_ImagesRead = true;

	for (size_t i = 0; i < detectThreads.size(); ++i)
		detectThreads[i].join();

	return 0;
}