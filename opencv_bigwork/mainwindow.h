#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <QMainWindow>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QTextCodec>
#include <QLabel>
#include <time.h>
#include "noisedialog.h"
#include "imagecropper.h"
#include "imagecropperdialog.h"
#include "noisedialog.h"
#include "filterdialog.h"
#include "concatdialog.h"
#include "facedialog.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    cv::Mat srcimgMat;
    cv::Mat dstimgMat;
    cv::Mat tempMat;
    QImage img;
    QString fileName;

    cv::Mat g_srcImage;
    cv::Mat g_grayImage;
    int g_nThresh = 80;
    int g_nThresh_max = 255;
    cv::RNG g_rng;
    cv::Mat g_cannyMat_output;
    vector<vector<cv::Point> > g_vContours;
    vector<cv::Vec4i> g_vHierarchy;
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
public:
    //操作
    QImage QCVMat2QImage(const cv::Mat&);
    void ShowImage();
    void ShowSrcImage();
    QImage ImageCenter(QImage, QLabel*);
    cv::Mat Contrast(cv::Mat src, int percent);
    cv::Mat Saturation(cv::Mat src, int percent);
    double dist(cv::Point a, cv::Point b);
    double getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center);
    void generateGradient(cv::Mat& mask, double power);
    cv::Mat loadFromQrc(QString qrc);
private slots:
    void on_actionopenfile_triggered();

    void on_pushButton_cut_clicked();

    void on_pushButton_noi_clicked();

    void on_pushButton_fliter_clicked();

    void on_pushButton_turnright_clicked();

    void on_pushButton_origin_clicked();

    void receiveMat(cv::Mat data){
        dstimgMat = data.clone();
        img=QCVMat2QImage(dstimgMat);
        ShowImage();
    }
    void on_pushButton_turnleft_clicked();

    void on_pushButton_mirrorY_clicked();

    void on_pushButton_mirrorX_clicked();

    void on_actionsavefile_triggered();

    void on_horizontalSlider_light_valueChanged(int value);

    void on_pushButton_choose_clicked();

    void on_horizontalSlider_contrast_valueChanged(int value);

    void on_pushButton_lightcomfirm_clicked();

    void on_pushButton_contrastcomfirm_clicked();

    void on_horizontalSlider_saturation_valueChanged(int value);

    void on_pushButton_satucomfirm_clicked();

    void on_pushButton_concat_clicked();

    void on_horizontalSlider_halation_valueChanged(int value);

    void on_pushButton_sumiao_clicked();

    void on_pushButton_cam_clicked();

    void on_horizontalSlider_contour_valueChanged(int value);

    void on_actiondock_show_triggered();

    void on_pushButton_face_clicked();

    void on_pushButton_edge_clicked();

    void on_pushButton_cam_2_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
