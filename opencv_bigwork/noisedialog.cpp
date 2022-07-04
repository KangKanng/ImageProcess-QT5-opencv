#include "noisedialog.h"
#include "ui_noisedialog.h"
#include "mainwindow.h"
NoiseDialog::NoiseDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::NoiseDialog)
{
    ui->setupUi(this);
}

NoiseDialog::~NoiseDialog()
{
    delete ui;
}

QImage NoiseDialog::QCVMat2QImage(const cv::Mat& mat)
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
QImage NoiseDialog::ImageCenter(QImage qimage, QLabel *qLabel)
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
void NoiseDialog::ShowImage()
{
    QImage Image=ImageCenter(img,ui->label_show);
    ui->label_show->setPixmap(QPixmap::fromImage(Image));
    ui->label_show->setAlignment(Qt::AlignCenter);
}
//高斯变换
void NoiseDialog::on_pushButton_gau_clicked()
{
    aa = AddGaussianNoise(a,ui->input_a->value(),ui->input_b->value());
    img = QCVMat2QImage(aa);
    ShowImage();
}
cv::Mat NoiseDialog::AddGaussianNoise(cv::Mat &srcImg,cv::InputArray a, cv::InputArray b)
{
    cv::Mat tempSrcImg = srcImg.clone();
    cv::Mat img_output(tempSrcImg.size(), tempSrcImg.type());
    //构造高斯噪声矩阵
    cv::Mat noise(tempSrcImg.size(), tempSrcImg.type());//创建一个噪声矩阵
    cv::RNG rng(time(NULL));
    rng.fill(noise, cv::RNG::NORMAL, a, b);//高斯分布；均值为10，标准差为36
    //将高斯噪声矩阵与原图像叠加得到含噪图像
    cv::add(tempSrcImg, noise, img_output);
    return img_output;
}
//给原图像增加椒盐噪声
//图象模拟添加椒盐噪声是通过随机获取像素点,并设置为高亮度点和低亮度点来实现的
//srcImage为源图像，n为椒/盐像素点个数，返回含噪图像
cv::Mat NoiseDialog::addSaltNoise(const cv::Mat srcImage, int n)
{
    cv::Mat dstImage = srcImage.clone();

    for (int k = 0; k < n; k++)
    {
        //随机取值行列，得到像素点(i,j)
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;

        //图像通道判定
        if (dstImage.channels() == 1)//修改像素点(i,j)的像素值
        {
            dstImage.at<uchar>(i, j) = 255;     //盐噪声
        }
        else
        {
            dstImage.at<cv::Vec3b>(i, j)[0] = 255;
            dstImage.at<cv::Vec3b>(i, j)[1] = 255;
            dstImage.at<cv::Vec3b>(i, j)[2] = 255;
        }
    }

    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 0;       //椒噪声
        }
        else
        {
            dstImage.at<cv::Vec3b>(i, j)[0] = 0;
            dstImage.at<cv::Vec3b>(i, j)[1] = 0;
            dstImage.at<cv::Vec3b>(i, j)[2] = 0;
        }
    }
    return dstImage;
}
void NoiseDialog::on_pushButton_2_clicked()
{
    aa = addSaltNoise(a,ui->input_n->value());
    img = QCVMat2QImage(aa);
    ShowImage();
}
cv::Mat NoiseDialog::AddRandomNoise(cv::Mat &srcImg, cv::InputArray a, cv::InputArray b)
{
    cv::Mat tempSrcImg = srcImg.clone();
    cv::Mat img_output(tempSrcImg.size(), tempSrcImg.type());
    //构造高斯噪声矩阵
    cv::Mat noise(tempSrcImg.size(), tempSrcImg.type());//创建一个噪声矩阵
    cv::RNG rng(time(NULL));
    rng.fill(noise, cv::RNG::UNIFORM, a, b);//高斯分布；均值为10，标准差为36
    //将高斯噪声矩阵与原图像叠加得到含噪图像
    cv::add(tempSrcImg, noise, img_output);
    return img_output;
}
void NoiseDialog::on_pushButton_3_clicked()
{
    aa = AddRandomNoise(a,ui->input_a_2->value(),ui->input_b_2->value());
    img = QCVMat2QImage(aa);
    ShowImage();
}

void NoiseDialog::on_buttonBox_accepted()
{
    sendMat(aa);
}
