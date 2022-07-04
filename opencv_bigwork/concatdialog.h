#ifndef CONCATDIALOG_H
#define CONCATDIALOG_H
#include <opencv2/opencv.hpp>
#include <QDialog>
#include <QLabel>
#include <QDebug>
#include <QFileDialog>
#include <vector>
#include <iostream>
#include <QTextCodec>
#include <QMessageBox>
using namespace std;
namespace Ui {
class ConcatDialog;
}

class ConcatDialog : public QDialog
{
    Q_OBJECT
public:
    cv::Mat src1;
    cv::Mat src2;
    cv::Mat dst;
    QImage img;
    QString filename1, filename2;
public:
    explicit ConcatDialog(QWidget *parent = nullptr);
    ~ConcatDialog();
    QImage QCVMat2QImage(const cv::Mat&);
    void ShowImage(QLabel*);
    QImage ImageCenter(QImage, QLabel*);
    bool ImageOverlap0(cv::Mat &img1, cv::Mat &img2);

signals:
    void sendMat(cv::Mat);
private slots:
    void on_pushButton_vertical_clicked();

    void on_pushButton_open1_clicked();

    void on_pushButton_open2_clicked();

    void on_pushButton_horizomal_clicked();

    void on_pushButton_model_clicked();

    void on_pushButton_SIFT_clicked();

    void on_pushButton_SURF_clicked();

    void on_pushButton_bend_clicked();

    void on_pushButton_fisheye_clicked();

    void on_buttonBox_accepted();

private:
    Ui::ConcatDialog *ui;
};

#endif // CONCATDIALOG_H
