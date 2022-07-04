#ifndef FACEDIALOG_H
#define FACEDIALOG_H

#include <QDialog>
#include <QString>
#include <QTextCodec>
#include <QDebug>
#include <QLabel>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/video.hpp"
#include <opencv2/face.hpp>
#include <opencv2/face/facemarkLBF.hpp>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
using namespace std;
namespace Ui {
class FaceDialog;
}

class FaceDialog : public QDialog
{
    Q_OBJECT
public:
    cv::Mat src;
    QImage img;
    cv::Mat dst;
    bool isDetected = false;
    cv::Mat m_MainImg;
    std::vector<std::vector<cv::Point2f>> m_vecFaceData;
    QString filename;
public:
    explicit FaceDialog(QWidget *parent = nullptr);
    ~FaceDialog();
    QImage QCVMat2QImage(const cv::Mat&);
    void ShowImage();
    cv::CascadeClassifier loadCascadeClassifier(const string cascadePath);
    std::vector<std::vector<cv::Point2f>> dectectFace68(QString filename);
    void detectAndDraw(cv::Mat& img, cv::CascadeClassifier& cascade,  double scale, int val);
    QImage ImageCenter(QImage qimage, QLabel *qLabel);
    void whiteFace(cv::Mat& matSelfPhoto,int alpha, int beta);
    void BilinearInsert(cv::Mat &src, cv::Mat &dst, float ux, float uy, int i, int j);
    cv::Mat LocalTranslationWarp_Face(cv::Mat &img, int warpX, int warpY, int endX, int endY, float radius);
    void LocalTranslationWarp_Eye(cv::Mat &img, cv::Mat &dst, int warpX, int warpY, int endX, int endY, float radius);
private slots:
    void on_horizontalSlider_wihte_valueChanged(int value);

    void on_horizontalSlider_wihte_2_valueChanged(int value);

    void on_pushButton_clicked();

    void on_horizontalSlider_2_valueChanged(int value);

    void on_horizontalSlider_3_valueChanged(int value);

private:
    Ui::FaceDialog *ui;
};

#endif // FACEDIALOG_H
