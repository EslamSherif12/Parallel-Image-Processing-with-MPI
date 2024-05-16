#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

// Function to distribute image data among processes
void distributeImage(const Mat& inputImage, Mat& localImage, int world_rank, int world_size) {
    int totalRows = inputImage.rows;
    int rowsPerProcess = totalRows / world_size;
    int remainingRows = totalRows % world_size;

    // Calculate the starting row index for the current process
    int startRow = world_rank * rowsPerProcess + min(world_rank, remainingRows);

    // Calculate the number of rows each process will receive
    int rowsToSend = (world_rank < remainingRows) ? rowsPerProcess + 1 : rowsPerProcess;

    // Define the ROI for the current process
    Rect roi(0, startRow, inputImage.cols, rowsToSend);

    // Extract the local image for the current process
    localImage = inputImage(roi).clone();
}
// Function to gather processed image parts from all processes
void gatherProcessedImages(const Mat& localImage, Mat& gatheredImage, int world_rank, int world_size) {
    // Allocate memory for the receive buffer on the root process
    if (world_rank == 0) {
        gatheredImage.create(localImage.rows * world_size, localImage.cols, localImage.type());
    }

    // Gather the processed parts from all processes
    MPI_Gather(localImage.data, localImage.total(), MPI_CHAR,
        gatheredImage.data, localImage.total(), MPI_CHAR,
        0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //string imagePath = "E:\\imigin.jpg", savedPath = "E:\\image outputtt.jpg";
    string imagePath = "E:\\imigin.jpg", savedPath = "image outputtt.jpg";

    Mat inputImage = imread(imagePath, IMREAD_GRAYSCALE);
    Mat localImage;

    // Distribute the color image among processes
    distributeImage(inputImage, localImage, world_rank, world_size);

    int choice;
    if (world_rank == 0) {
        cout << "\t\t\t*\n";
        cout << "\t\t\tWelcome to Parallel Image Processing with MPI\n";
        cout << "\t\t\t*\n\n\n";
        cout << "Please choose an image processing operation:\n";
        cout << "01- Gaussian Blur\n";
        cout << "02- Edge Detection\n";
        cout << "03- Image Rotation\n";
        cout << "04- Image Scaling\n";
        cout << "05- Histogram Equalization\n";
        cout << "06- Color Space Conversion\n";
        cout << "07- Global Thresholding\n";
        cout << "08- Local Thresholding\n";
        cout << "09- Image Compression\n";
        cout << "10- Median\n";
        cout << "\nEnter your choice (1-10): ";
        string choiceStr;
        cin >> choiceStr;
        choice = stoi(choiceStr);
    }

    // Broadcast user choice to all processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (choice == 1)
    {
        // Gaussian Blur
        int blurRadius = 9;
        if (world_rank == 0) {
            cout << "\nYou have selected Gaussian Blur.\n\n";
            cout << "Please enter the blur radius: ";
            cin >> blurRadius;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat blurredImage;
        GaussianBlur(localImage, blurredImage, Size(blurRadius, blurRadius), 0);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Gaussian Blur...\n\n";
            cout << "Gaussian Blur operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Blurred image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(blurredImage, inputImage, world_rank, world_size);
    }
    if (choice == 2) {
        // Edge Detection
        if (world_rank == 0) {
            cout << "You have selected Edge Detection.\n";
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat edgeImage;

        if (localImage.channels() > 1) {
            cvtColor(localImage, edgeImage, COLOR_BGR2GRAY);
        }
        else {
            edgeImage = localImage.clone();
        }

        Canny(edgeImage, edgeImage, 100, 200);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Edge Detection...\n\n";
            cout << "Edge Detection operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(edgeImage, inputImage, world_rank, world_size);
    }
    if (choice == 3) {
        // Image Rotation
        double angle = 45;
        if (world_rank == 0) {
            cout << "You have selected Image Rotation.\n";
            cout << "Please enter the rotation angle: ";
            cin >> angle;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat rotatedImage;

        Point2f center(localImage.cols / 2.0, localImage.rows / 2.0);
        Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(inputImage, rotatedImage, rotationMatrix, inputImage.size());
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Image Rotation...\n\n";
            cout << "Image Rotation operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(rotatedImage, inputImage, world_rank, world_size);
    }
    else if (choice == 4) {
        // Image Scaling
        double scaleX = 1.5;
        double scaleY = 1.5;
        if (world_rank == 0) {
            cout << "You have selected Image Scaling.\n";
            cout << "Please enter the scaling factor along X-axis: ";
            cin >> scaleX;
            cout << "Please enter the scaling factor along Y-axis: ";
            cin >> scaleY;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat scaledImage;
        resize(localImage, scaledImage, Size(), scaleX, scaleY);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Image Scaling...\n\n";
            cout << "Image Scaling operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(scaledImage, inputImage, world_rank, world_size);
    }
    else if (choice == 5) {
        // Histogram Equalization
        if (world_rank == 0) {
            cout << "You have selected Histogram Equalization.\n";
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat equalizedImage;
        if (localImage.channels() > 1) {
            cvtColor(localImage, equalizedImage, COLOR_BGR2GRAY);
        }
        else {
            equalizedImage = localImage.clone(); // If already grayscale, no need to convert
        }
        equalizeHist(equalizedImage, equalizedImage);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Histogram Equalization...\n\n";
            cout << "Histogram Equalization operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(equalizedImage, inputImage, world_rank, world_size);
    }
    else if (choice == 6) {
        // Color Space Conversion
        int code = COLOR_BGR2GRAY;
        if (world_rank == 0) {
            cout << "You have selected Color Space Conversion.\n";
            cout << "Please enter the color space conversion code (e.g., CV_BGR2GRAY = 6 , CV_GRAY2BGR = 8): ";
            cin >> code;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat convertedImage;
        if (localImage.channels() > 1) {
            cvtColor(localImage, convertedImage, code);
        }
        else {
            convertedImage = localImage.clone(); // If already grayscale, no need to convert
        }
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Color Space Conversion...\n\n";
            cout << "Color Space Conversion operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(convertedImage, inputImage, world_rank, world_size);
    }
    else if (choice == 7) {
        // Global Thresholding
        int thresholdValue = 128;
        if (world_rank == 0) {
            cout << "You have selected Global Thresholding.\n";
            cout << "Please enter the threshold value: ";
            cin >> thresholdValue;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat thresholdedImage;
        if (localImage.channels() > 1) {
            cvtColor(localImage, thresholdedImage, COLOR_BGR2GRAY);
        }
        else {
            thresholdedImage = localImage.clone();
        }
        threshold(thresholdedImage, thresholdedImage, thresholdValue, 255, THRESH_BINARY);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Global Thresholding...\n\n";
            cout << "Global Thresholding operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(thresholdedImage, inputImage, world_rank, world_size);
    }
    else if (choice == 8) {
        // Local Thresholding
        int blockSize = 11;
        double constant = 2;
        if (world_rank == 0) {
            cout << "You have selected Local Thresholding.\n";
            cout << "Please enter the block size: ";
            cin >> blockSize;
            cout << "Please enter the constant: ";
            cin >> constant;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat thresholdedImage;
        if (localImage.channels() > 1) {
            cvtColor(localImage, thresholdedImage, COLOR_BGR2GRAY);
        }
        else {
            thresholdedImage = localImage.clone();
        }

        adaptiveThreshold(thresholdedImage, thresholdedImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, constant);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Local Thresholding...\n\n";
            cout << "Local Thresholdingn operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(thresholdedImage, inputImage, world_rank, world_size);
    }
    else if (choice == 9) {
        // Image Compression
        int compressionFactor = 70;
        if (world_rank == 0) {
            cout << "You have selected Image Compression.\n";
            cout << "Please enter the compression level (0-100): ";
            cin >> compressionFactor;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat compressedImage;
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_JPEG_QUALITY);
        compression_params.push_back(compressionFactor);

        vector<uchar> encodedImage;
        imencode(".jpg", localImage, encodedImage, compression_params);

        Mat outputImage = imdecode(encodedImage, IMREAD_COLOR);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Image Compression...\n\n";
            cout << "Image Compression operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(outputImage, inputImage, world_rank, world_size);
    }

    if (choice == 10) {
        // Median Filtering
        int kernelSize = 5;
        if (world_rank == 0) {
            cout << "You have selected Median Filtering.\n";
            cout << "Please enter the kernel size (odd number): ";
            cin >> kernelSize;
        }
        auto start = chrono::steady_clock::now(); // Start the timer
        Mat medianImage;
        medianBlur(localImage, medianImage, kernelSize);
        auto end = chrono::steady_clock::now(); // Stop the timer
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count(); // Calculate the duration
        if (world_rank == 0) {
            cout << "\nProcessing image " << imagePath << " with Median Filtering...\n\n";
            cout << "Median Filtering operation completed successfully in " << (duration * 0.001) << " seconds.\n\n";
            cout << "Converted image saved as " << savedPath << ".\n";
        }
        gatherProcessedImages(medianImage, inputImage, world_rank, world_size);
    }

    if (world_rank == 0) {
        imwrite(savedPath, inputImage);
    }

    MPI_Finalize();

    if (world_rank == 0) {
        cout << "\nThank you for using Parallel Image Processing with MPI.\n\n";
    }

    return 0;
}