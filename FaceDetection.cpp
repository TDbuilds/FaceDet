#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/objdetect.hpp"

#include <iostream>

using namespace cv;

int main()
{
    //###########################################################################################
    //change mode for detection
    int mode = 0; // 0 = haarcascades, 1 = YuNet 
    //###########################################################################################

    Mat frame;
    VideoCapture cap;

    //open default -> webcam
    cap.open(0);

    CascadeClassifier classifier;

    //load Cascade Classifier
    if( !classifier.load(samples::findFileOrKeep("C://opencv//sources//data//haarcascades//haarcascade_frontalface_default.xml")))return 0;

    //path to model file
    cv::String modelPath = "C://opencv//models//FaceDetection//face_detection_yunet_2023mar_int8bq.onnx";

    int backendID = 0; //0: default, 1: Halide, 2: Intel's Inference Engine, 3: OpenCV, 4: VKCOM, 5: CUDA
    int targetID = 0; //0: CPU, 1: OpenCL, 2: OpenCL FP16, 3: Myriad, 4: Vulkan, 5: FPGA, 6: CUDA, 7: CUDA FP16, 8: HDDL

    float scoreThreshold = 0.7; //Threshold for fitering out 
    float nms_Threshold = 0.3; //Non-Max-Supression, find the best bounding box
    int topK = 1000; //keep bounding boxes before doing nms

    // Initialize FaceDetectorYN
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), scoreThreshold, nms_Threshold, topK, backendID, targetID);
  
    std::vector<Rect> foundFaces;

    cv::namedWindow("Webcam", WINDOW_NORMAL);

    while (cap.isOpened()) {

        //retrieve next frame
        cap >> frame;
        
        //detection with Cascades
        if (mode == 0) {
            //convert to grayscale image for better detection            
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            classifier.detectMultiScale(
                gray,
                foundFaces,
                1.2,
                2  ,
                0 | CASCADE_SCALE_IMAGE,
                Size(40,40)
                );

            for (auto face : foundFaces) {
                rectangle(
                    frame,
                    Point(face.x, face.y),
                    Point(face.x + face.width, face.y + face.height),
                    Scalar(255,0,100),
                    1,
                    LINE_8,
                    0
                );
            }
        }
        //detection with YuNet
        if (mode == 1) {

            Mat faces;

            int height = frame.size().height;
            int width = frame.size().width;

            detector->setInputSize(Size(width, height));

            //detects faces and saves them in Mat
            detector->detect(frame, faces);

            for (int i = 0; i < faces.rows; i++) {

                //get bounding Box Data from detected Faces
                int topLx = faces.at<float>(i, 0);
                int topLy = faces.at<float>(i, 1);

                int box_width = faces.at<float>(i, 2);
                int box_height = faces.at<float>(i, 3);

                // display the confidence
                cv::putText(frame, cv::format("%.4f", faces.at<float>(i, 14)), cv::Point2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)) + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

                rectangle(
                    frame,
                    Point(topLx, topLy),
                    Point(topLx + box_width, topLy + box_height),
                    Scalar(0, 0, 255),
                    1,
                    LINE_8,
                    0
                );
            }
        }

        imshow("Webcam", frame);


        //break with key input
        if (waitKey(1) >= 0)
        {
            cap.release();
            break;
        }
    }

    return 0;
}
