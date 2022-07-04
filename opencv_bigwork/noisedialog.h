#ifndef NOISEDIALOG_H
#define NOISEDIALOG_H
#include <opencv2/opencv.hpp>
#include <QLabel>
#include <QDialog>
#include <QDebug>

namespace Ui {
class NoiseDialog;
}

class NoiseDialog : public QDialog
{
    Q_OBJECT
public:
    cv::Mat a;
    cv::Mat aa;
    QImage img;
public:
    explicit NoiseDialog(QWidget *parent = nullptr);
    ~NoiseDialog();
    void get(cv::Mat b){a=b.clone();}
    QImage QCVMat2QImage(const cv::Mat&);
    void ShowImage();
    QImage ImageCenter(QImage, QLabel*);
    cv::Mat addSaltNoise(const cv::Mat srcImage, int n);
    cv::Mat AddGaussianNoise(cv::Mat &srcImg, cv::InputArray a, cv::InputArray b);
    cv::Mat AddRandomNoise(cv::Mat &srcImg, cv::InputArray a, cv::InputArray b);
private slots:
    void on_pushButton_gau_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_buttonBox_accepted();
signals:
    void sendMat(cv::Mat);
private:
    Ui::NoiseDialog *ui;
};

#endif // NOISEDIALOG_H
