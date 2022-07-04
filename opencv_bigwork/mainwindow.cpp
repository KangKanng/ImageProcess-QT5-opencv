#include "mainwindow.h"
#include "ui_mainwindow.h"
#define max2(a,b) (a>b?a:b)
#define max3(a,b,c) (a>b?max2(a,c):max2(b,c))
#define min2(a,b) (a<b?a:b)
#define min3(a,b,c) (a<b?min2(a,c):min2(b,c))
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //设置页面的标题和图标
    this->setWindowTitle("数字图像处理系统");
    this->setWindowIcon(QIcon(":/images/picture.png"));
    setStyleSheet(tr("background-image: url(:/images/bg.jpg)"));
}

MainWindow::~MainWindow()
{
    delete ui;
}

//菜单
//打开文件
void MainWindow::on_actionopenfile_triggered()
{
    fileName = QFileDialog::getOpenFileName(this, "Open Image", ".", "Image Files(*.png *.jpg *.jpeg *.bmp)");
    if(!fileName.isEmpty())
    {
        QTextCodec *code = QTextCodec::codecForName("gb2312");
        std::string name = code->fromUnicode(fileName).data();
        srcimgMat = cv::imread(name);
        img = QCVMat2QImage(srcimgMat);
        dstimgMat = srcimgMat.clone();
        ShowImage();
    }
}
//保存图像
void MainWindow::on_actionsavefile_triggered()
{
    if(!img.isNull())
    {
        QString filename = QFileDialog::getSaveFileName(this,tr("保存图片"), "",
        tr("*.png;; *.jpg;; *.bmp;; *.tif;; *.GIF")); //选择路径
        if (filename.isEmpty()) return;
        else
        {
            if (!(img.save(filename))) //保存图像
            {
                QMessageBox::information(this,tr("图片保存成功！"),tr("图片保存失败！"));
                return;
            }
            ui->statusBar->showMessage("图片保存成功！");
        }

    }
    else
    {
        QMessageBox::warning(nullptr, "提示", "请先打开图片！", QMessageBox::Yes |  QMessageBox::Yes);
    }
}
void MainWindow::on_actiondock_show_triggered()
{
    ui->dockWidget_1->show();
    ui->dockWidget_2->show();
}

//dialogs
//裁剪dialog
void MainWindow::on_pushButton_cut_clicked()
{
    ImageCropperDemo* dialog = new ImageCropperDemo(this);
    dialog->filename = fileName;
    dialog->show();
    dialog->onChooseOriginalImage();
}
//噪声dialog
void MainWindow::on_pushButton_noi_clicked()
{
    NoiseDialog* dialog = new NoiseDialog(this);
    connect(dialog ,SIGNAL(sendMat(cv::Mat)), this,SLOT(receiveMat(cv::Mat)));
    dialog->show();
    dialog->get(dstimgMat);
    dialog->aa = dialog->a.clone();
    dialog->img = dialog->QCVMat2QImage(dialog->a);
    dialog->ShowImage();
}
//滤波dialog
void MainWindow::on_pushButton_fliter_clicked()
{
    FilterDialog* dialog = new FilterDialog(this);
    connect(dialog ,SIGNAL(sendMat(cv::Mat)), this,SLOT(receiveMat(cv::Mat)));
    dialog->show();
    dialog->get(dstimgMat);
    dialog->img = dialog->QCVMat2QImage(dialog->src);
    dialog->ShowImage();
}
//拼接dialog
void MainWindow::on_pushButton_concat_clicked()
{
    ConcatDialog* dialog = new ConcatDialog(this);
    connect(dialog ,SIGNAL(sendMat(cv::Mat)), this,SLOT(receiveMat(cv::Mat)));
    dialog->show();
}
//人脸dialog
void MainWindow::on_pushButton_face_clicked()
{
    FaceDialog* dialog = new FaceDialog(this);
    dialog->src = dstimgMat.clone();
    dialog->filename = fileName;
    connect(dialog ,SIGNAL(sendMat(cv::Mat)), this,SLOT(receiveMat(cv::Mat)));
    dialog->show();
    dialog->img = dialog->QCVMat2QImage(dialog->src);
    dialog->ShowImage();
    dialog->m_vecFaceData = dialog->dectectFace68(fileName);
}


//按钮
//向右旋转
void MainWindow::on_pushButton_turnright_clicked()
{
    cv::rotate(dstimgMat,dstimgMat,0);
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//重置
void MainWindow::on_pushButton_origin_clicked()
{
    ShowSrcImage();
}
//向左旋转
void MainWindow::on_pushButton_turnleft_clicked()
{
    cv::rotate(dstimgMat,dstimgMat,2);
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//Y轴翻转
void MainWindow::on_pushButton_mirrorY_clicked()
{
    cv::flip(dstimgMat,dstimgMat,0);
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//X轴翻转
void MainWindow::on_pushButton_mirrorX_clicked()
{
    cv::flip(dstimgMat,dstimgMat,1);
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//亮度确定
void MainWindow::on_pushButton_lightcomfirm_clicked()
{
    dstimgMat = tempMat.clone();
    ui->horizontalSlider_light->setValue(0);
}
//对比度确定
void MainWindow::on_pushButton_contrastcomfirm_clicked()
{
    dstimgMat = tempMat.clone();
    ui->horizontalSlider_contrast->setValue(0);
}
//饱和度确定
void MainWindow::on_pushButton_satucomfirm_clicked()
{
    dstimgMat = tempMat.clone();
    ui->horizontalSlider_saturation->setValue(0);
}
//素描效果
void MainWindow::on_pushButton_sumiao_clicked()
{
    cv::Mat gray;
    cvtColor(dstimgMat,gray,cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    GaussianBlur(gray, blur, cv::Size(15, 15), 0, 0);

    // 提取纹理
    cv::Mat veins;
    divide(gray, blur, veins, 255);

    // 加深处理
    cv::Mat deepenb, deepena;
    divide(255-veins, blur, deepenb, 255);
    deepena = 255 - deepenb;

    dstimgMat = deepena;
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//摄像头
void MainWindow::on_pushButton_cam_clicked()
{
    cv::VideoCapture vCap;
    vCap.open(0); // 打开摄像头 ,cv::CAP_V4L
    if(!vCap.isOpened()){
        std::cout << "摄像头打开失败" << std::endl;
        return;
    }
    cv::Mat frames;
    bool ret=0;
    int keyV=0;
    while(1){
        ret = vCap.read(frames);
        if(!ret){
            std::cout << "read cap fail" << std::endl;
            continue;
        }
        cv::imshow("cap", frames);
        keyV = cv::waitKey(20);
        if (keyV == ' ') {
            srcimgMat = frames;
            dstimgMat = frames;
            img = QCVMat2QImage(srcimgMat);
            ShowImage();
            break;
        }
        if (keyV == 27) {break;}
       }
    vCap.release();
    cv::destroyAllWindows();

}


//滑动条
//亮度
void MainWindow::on_horizontalSlider_light_valueChanged(int value)
{
    dstimgMat.convertTo(tempMat, -1, 1, value);
    img = QCVMat2QImage(tempMat);
    ShowImage();
}
//对比度
void MainWindow::on_horizontalSlider_contrast_valueChanged(int value)
{
    tempMat = Contrast(dstimgMat,value);
    img = QCVMat2QImage(tempMat);
    ShowImage();
}
//饱和度
void MainWindow::on_horizontalSlider_saturation_valueChanged(int value)
{
    tempMat = Saturation(dstimgMat, value);
    img = QCVMat2QImage(tempMat);
    ShowImage();
}
//晕影
void MainWindow::on_horizontalSlider_halation_valueChanged(int value)
{
    cv::Mat mask_img(dstimgMat.size(), CV_64F);
    generateGradient(mask_img, value/40);

    cv::Mat gradient;
    cv::normalize(mask_img, gradient, 0, 255, 32); //cv::normalize(maskImg, gradient, 0, 255, CV_MINMAX);
    //cv::imwrite("gradient.png", gradient);

    cv::Mat lab_img(dstimgMat.size(), CV_8UC3);
    cv::cvtColor(dstimgMat, lab_img, cv::COLOR_BGR2Lab);

    for (int row = 0; row < lab_img.size().height; row++)
    {
        for (int col = 0; col < lab_img.size().width; col++)
        {
            cv::Vec3b value = lab_img.at<cv::Vec3b>(row, col);
            value.val[0] *= mask_img.at<double>(row, col);
            lab_img.at<cv::Vec3b>(row, col) = value;
         }
    }

    cv::Mat output;
    cv::cvtColor(lab_img, output, cv::COLOR_Lab2BGR);
    tempMat = output;
    img = QCVMat2QImage(tempMat);
    ShowImage();
}
//描边Canny算子
void MainWindow::on_horizontalSlider_contour_valueChanged(int value)
{
    g_nThresh = value;
    g_srcImage = dstimgMat.clone();
    cvtColor(g_srcImage, g_grayImage, cv::COLOR_BGR2GRAY);
    blur(g_grayImage, g_grayImage, cv::Size(3, 3));
    Canny(g_grayImage, g_cannyMat_output, g_nThresh, g_nThresh * 2, 3);

    findContours(g_cannyMat_output, g_vContours, g_vHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::Mat drawing = cv::Mat::zeros(g_cannyMat_output.size(), CV_8UC3);
    for (int i = 0; i < g_vContours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
        drawContours(drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, cv::Point());
    }
    tempMat = drawing;
    img = QCVMat2QImage(tempMat);
    ShowImage();
}


//操作operations
//将cv::Mat转换为QImage
QImage MainWindow::QCVMat2QImage(const cv::Mat& mat)
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
//图片居中显示,图片大小与label大小相适应
QImage MainWindow::ImageCenter(QImage qimage, QLabel *qLabel)
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
//显示图片
void MainWindow::ShowImage()
{
    QImage Image=ImageCenter(img,ui->label_show);
    ui->label_show->setPixmap(QPixmap::fromImage(Image));
    ui->label_show->setAlignment(Qt::AlignCenter);
}
//显示原图
void MainWindow::ShowSrcImage()
{
    dstimgMat = srcimgMat.clone();
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
//打开图片
void MainWindow::on_pushButton_choose_clicked()
{
    fileName = QFileDialog::getOpenFileName(this, "Open Image", ".", "Image Files(*.png *.jpg *.jpeg *.bmp)");
    if(!fileName.isEmpty())
    {
        QTextCodec *code = QTextCodec::codecForName("gb2312");
        std::string name = code->fromUnicode(fileName).data();
        srcimgMat = cv::imread(name);
        img = QCVMat2QImage(srcimgMat);
        dstimgMat = srcimgMat.clone();
        ShowImage();
    }
}
//对比度
cv::Mat MainWindow::Contrast(cv::Mat src, int percent)
{
    float alpha = percent / 100.f;
    alpha = max(-1.f, min(1.f, alpha));
    cv::Mat temp = src.clone();
    int row = src.rows;
    int col = src.cols;
    int thresh = 127;
    for (int i = 0; i < row; ++i)
    {
        uchar* t = temp.ptr<uchar>(i);
        uchar* s = src.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            uchar b = s[3 * j];
            uchar g = s[3 * j + 1];
            uchar r = s[3 * j + 2];
            int newb, newg, newr;
            if (alpha == 1)
            {
                t[3 * j + 2] = r > thresh ? 255 : 0;
                t[3 * j + 1] = g > thresh ? 255 : 0;
                t[3 * j] = b > thresh ? 255 : 0;
                continue;
            }
            else if (alpha >= 0)
            {
                newr = static_cast<int>(thresh + (r - thresh) / (1 - alpha));
                newg = static_cast<int>(thresh + (g - thresh) / (1 - alpha));
                newb = static_cast<int>(thresh + (b - thresh) / (1 - alpha));
            }
            else {
                newr = static_cast<int>(thresh + (r - thresh) * (1 + alpha));
                newg = static_cast<int>(thresh + (g - thresh) * (1 + alpha));
                newb = static_cast<int>(thresh + (b - thresh) * (1 + alpha));

            }
            newr = max(0, min(255, newr));
            newg = max(0, min(255, newg));
            newb = max(0, min(255, newb));
            t[3 * j + 2] = static_cast<uchar>(newr);
            t[3 * j + 1] = static_cast<uchar>(newg);
            t[3 * j] = static_cast<uchar>(newb);
        }
    }
    return temp;
}
//饱和度
cv::Mat MainWindow::Saturation(cv::Mat src, int percent)
{
    float Increment = percent * 1.0f / 100;
    cv::Mat temp = src.clone();
    int row = src.rows;
    int col = src.cols;
    for (int i = 0; i < row; ++i)
    {
        uchar* t = temp.ptr<uchar>(i);
        uchar* s = src.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            uchar b = s[3 * j];
            uchar g = s[3 * j + 1];
            uchar r = s[3 * j + 2];
            float max = max3(r, g, b);
            float min = min3(r, g, b);
            float delta, value;
            float L, S, alpha;
            delta = (max - min) / 255;
            if (delta == 0)
                continue;
            value = (max + min) / 255;
            L = value / 2;
            if (L < 0.5)
                S = delta / value;
            else
                S = delta / (2 - value);
            if (Increment >= 0)
            {
                if ((Increment + S) >= 1)
                    alpha = S;
                else
                    alpha = 1 - Increment;
                alpha = 1 / alpha - 1;
                t[3 * j + 2] = static_cast<uchar>(r + (r - L * 255) * alpha);
                t[3 * j + 1] = static_cast<uchar>(g + (g - L * 255) * alpha);
                t[3 * j] = static_cast<uchar>(b + (b - L * 255) * alpha);
            }
            else
            {
                alpha = Increment;
                t[3 * j + 2] = static_cast<uchar>(L * 255 + (r - L * 255) * (1 + alpha));
                t[3 * j + 1] = static_cast<uchar>(L * 255 + (g - L * 255) * (1 + alpha));
                t[3 * j] = static_cast<uchar>(L * 255 + (b - L * 255) * (1 + alpha));
            }
        }
    }
    return temp;
}
//晕影
double MainWindow::dist(cv::Point a, cv::Point b)
{
    return sqrt(pow((double)(a.x - b.x), 2) + pow((double)(a.y - b.y), 2));
}
//用于计算从边缘到中心点最远的距离。
double MainWindow::getMaxDisFromCorners(const cv::Size& imgSize, const cv::Point& center)
{
    // given a rect and a line | 给定一个矩形和一条线
    // get which corner of rect is farthest from the line | 得到哪个角的矩形是离线最远


    std::vector<cv::Point> corners(4);
    corners[0] = cv::Point(0, 0);
    corners[1] = cv::Point(imgSize.width, 0);
    corners[2] = cv::Point(0, imgSize.height);
    corners[3] = cv::Point(imgSize.width, imgSize.height);

    double max_dis = 0;
    for (int i = 0; i < 4; ++i)
    {
        double dis = dist(corners[i], center);
        if (max_dis < dis)
            max_dis = dis;
    }

    return max_dis;
}
//助函数用于创建一个渐变的图像
//半径和功率是控制滤波器艺术效果的变量。
void MainWindow::generateGradient(cv::Mat& mask, double power)
{
    cv::Point first_point = cv::Point(mask.size().width / 2, mask.size().height / 2);
    double radius = 1.0;

    // max image radian | 最大图像半径
    double max_image_rad = radius * getMaxDisFromCorners(mask.size(), first_point);

    mask.setTo(cv::Scalar(1));
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double temp = dist(first_point, cv::Point(j, i)) / max_image_rad;
            temp = temp * power;
            double temp_s = pow(cos(temp), 4);
            mask.at<double>(i, j) = temp_s;
        }
    }
}

void MainWindow::on_pushButton_edge_clicked()
{
    cv::Mat edge;
    edge = loadFromQrc("./inputimg/edge.png");
    cv::resize(edge,edge,dstimgMat.size(),0,0,cv::INTER_CUBIC);
    addWeighted(edge,0.7,dstimgMat,0.3,0,dstimgMat);
    img = QCVMat2QImage(dstimgMat);
    ShowImage();
}
cv::Mat MainWindow::loadFromQrc(QString qrc)
{
    int flag = cv::IMREAD_COLOR;
    QFile file(qrc);
    cv::Mat m;
    if(file.open(QIODevice::ReadOnly)) {
        qint64 sz = file.size();
        std::vector<uchar> buf(sz);
        file.read((char*)buf.data(), sz);
        m = cv::imdecode(buf, flag);
    }

    return m;
}

void MainWindow::on_pushButton_cam_2_clicked()
{
    on_actionsavefile_triggered();
}
