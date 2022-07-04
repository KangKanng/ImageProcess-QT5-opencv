#ifndef FILTERDIALOG_H
#define FILTERDIALOG_H

#include <QDialog>
#include <QLabel>
#include <opencv2/opencv.hpp>
#include <QDebug>
namespace Ui {
class FilterDialog;
}

class FilterDialog : public QDialog
{
    Q_OBJECT
public:
    cv::Mat src;
    cv::Mat dst;
    QImage img;
public:
    explicit FilterDialog(QWidget *parent = nullptr);
    ~FilterDialog();
    void get(cv::Mat b){ src = b.clone();}
    QImage QCVMat2QImage(const cv::Mat&);
    void ShowImage();
    QImage ImageCenter(QImage, QLabel*);
signals:
    void sendMat(cv::Mat);
private slots:
    void on_pushButton_gauss_blur_clicked();

    void on_pushButton_double_blur_clicked();

    void on_pushButton_mean_blur_clicked();

    void on_pushButton_median_blur_clicked();

    void on_buttonBox_accepted();

private:
    Ui::FilterDialog *ui;
};

#endif // FILTERDIALOG_H
