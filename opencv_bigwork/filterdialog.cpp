#include "filterdialog.h"
#include "ui_filterdialog.h"

FilterDialog::FilterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FilterDialog)
{
    ui->setupUi(this);
}

FilterDialog::~FilterDialog()
{
    delete ui;
}
QImage FilterDialog::QCVMat2QImage(const cv::Mat& mat)
{
    const unsigned char* data = mat.data;

    int width = mat.cols;
    int height = mat.rows;
    int bytesPerLine = static_cast<int>(mat.step);
    switch(mat.type())
    {
        //8 bit , ARGB
        case CV_8UC4:
        {
            QImage image(data, width, height, bytesPerLine, QImage::Format_ARGB32);
            return image;
        }

        //8 bit BGR
        case CV_8UC3:
        {
            QImage image(data, width, height, bytesPerLine, QImage::Format_RGB888);
            //swap blue and red channel
            return image.rgbSwapped();
        }

        //8 bit Gray shale
        case CV_8UC1:
        {
            QImage image(data, width, height, bytesPerLine, QImage::Format_Grayscale8);
            return image;
        }

        //
        default:
        {
            //Unsupported format
            qWarning()<<"Unsupported cv::Mat type:"<<mat.type()
                     <<", Empty QImage will be returned!";
            return QImage();
        }
    }
}
QImage FilterDialog::ImageCenter(QImage qimage, QLabel *qLabel)
{
    QImage image;
    QSize imageSize = qimage.size();
    QSize labelSize = qLabel->size();

    double dWidthRatio = 1.0*imageSize.width() / labelSize.width();
    double dHeightRatio = 1.0*imageSize.height() / labelSize.height();
    if (dWidthRatio>dHeightRatio)
    {
        image = qimage.scaledToWidth(labelSize.width());
    }
    else
    {
        image = qimage.scaledToHeight(labelSize.height());
    }
    return image;
}
void FilterDialog::ShowImage()
{
    QImage Image=ImageCenter(img,ui->label_show);
    ui->label_show->setPixmap(QPixmap::fromImage(Image));
    ui->label_show->setAlignment(Qt::AlignCenter);
}

//高斯滤波
void FilterDialog::on_pushButton_gauss_blur_clicked()
{
    cv::GaussianBlur(src, dst, cv::Size(ui->spinBox_gauss_blur->value(),ui->spinBox_gauss_blur->value()), 0, 0);
    img = QCVMat2QImage(dst);
    ShowImage();
}
//双边
void FilterDialog::on_pushButton_double_blur_clicked()
{
    cv::bilateralFilter(src, dst, 25, 25*2, 25/2);
    img = QCVMat2QImage(dst);
    ShowImage();
}
//均值滤波
void FilterDialog::on_pushButton_mean_blur_clicked()
{
    cv::blur(src,dst,cv::Size(ui->spinBox_mean->value(),ui->spinBox_mean->value()));
    img = QCVMat2QImage(dst);
    ShowImage();
}
//中值滤波
void FilterDialog::on_pushButton_median_blur_clicked()
{
    cv::medianBlur(src,dst,ui->spinBox_median->value());
    img = QCVMat2QImage(dst);
    ShowImage();
}

void FilterDialog::on_buttonBox_accepted()
{
    sendMat(dst);
}
