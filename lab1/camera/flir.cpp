#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	// Retrieve singleton reference to system object
	SystemPtr system = System::GetInstance();

	// Retrieve list of cameras from the system
	CameraList camList = system->GetCameras();
	const unsigned int numCameras = camList.GetSize();
	cout << "Number of cameras detected: " << numCameras << endl << endl;

	// Finish if there are no cameras
	if (numCameras == 0)
	{
		// Clear camera list before releasing system
		camList.Clear();
		// Release system
		system->ReleaseInstance();
		cout << "Not enough cameras!" << endl;
		cout << "Done! Press Enter to exit..." << endl;
		getchar();
		return -1;
	}

	// Create shared pointer to camera
	//
	// *** NOTES ***
	// The CameraPtr object is a shared pointer, and will generally clean itself
	// up upon exiting its scope. However, if a shared pointer is created in the
	// same scope that a system object is explicitly released (i.e. this scope),
	// the reference to the shared point must be broken manually.
	//
	// *** LATER ***
	// Shared pointers can be terminated manually by assigning them to nullptr.
	// This keeps releasing the system from throwing an exception.
	//
	CameraPtr pCam = nullptr;
	int result = 0;
	Mat frame;
	// Select camera
	pCam = camList.GetByIndex(0);
	try
	{
		// Initialize camera
		pCam->Init();
		// Begin acquiring images
		//
		// *** NOTES ***
		// What happens when the camera begins acquiring images depends on the
		// acquisition mode. Single frame captures only a single image, multi
		// frame captures a set number of images, and continuous captures a
		// continuous stream of images. Because the example calls for the
		// retrieval of 10 images, continuous mode has been set.
		//
		// *** LATER ***
		// Image acquisition must be ended when no more images are needed.
		//
		pCam->BeginAcquisition();
		char c;
		ImagePtr pResultImage;
		while (c = waitKey(1) && pCam->IsStreaming())
		{
			pResultImage = pCam->GetNextImage();
			//
				// Ensure image completion
				//
				// *** NOTES ***
				// Images can easily be checked for completion. This should be
				// done whenever a complete image is expected or required.
				// Further, check image status for a little more insight into
				// why an image is incomplete.
				//
			if (pResultImage->IsIncomplete())
			{
				// Retrieve and print the image status description
				cout << "Image incomplete: " << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
					<< "..." << endl
					<< endl;
			}
			else
			{
				// Convert image to mono 8
				//
				// *** NOTES ***
				// Images can be converted between pixel formats by using
				// the appropriate enumeration value. Unlike the original
				// image, the converted one does not need to be released as
				// it does not affect the camera buffer.
				//
				// When converting images, color processing algorithm is an
				// optional parameter.
				//
				ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);
				
				frame = cv::Mat(convertedImage->GetHeight(), convertedImage->GetWidth(), CV_8UC1,
					(uint8_t *)convertedImage->GetData());
				/// show
				resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
				imshow("OpenCV Display Window", frame);
			}
		}
		//
		// Release image
		//
		// *** NOTES ***
		// Images retrieved directly from the camera (i.e. non-converted
		// images) need to be released in order to keep from filling the
		// buffer.
		//
		pResultImage->Release();
		// End acquiring images
		pCam->EndAcquisition();
		// Deinitialize camera
		pCam->DeInit();
	}
	catch (Spinnaker::Exception& e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	// Release reference to the camera
	//
	// *** NOTES ***
	// Had the CameraPtr object been created within the for-loop, it would not
	// be necessary to manually break the reference because the shared pointer
	// would have automatically cleaned itself up upon exiting the loop.
	//
	pCam = nullptr;
	// Clear camera list before releasing system
	camList.Clear();
	// Release system
	system->ReleaseInstance();
	cout << endl << "Done! Press Enter to exit..." << endl;
	getchar();

	return result;
}