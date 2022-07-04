#include "facedialog.h"
#include "ui_facedialog.h"
using namespace dlib;
using namespace std;
FaceDialog::FaceDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FaceDialog)
{
    ui->setupUi(this);
}

FaceDialog::~FaceDialog()
{
    delete ui;
}
QImage FaceDialog::QCVMat2QImage(const cv::Mat& mat)
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
QImage FaceDialog::ImageCenter(QImage qimage, QLabel *qLabel)
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
void FaceDialog::ShowImage()
{
    QImage Image=ImageCenter(img,ui->label_show);
    ui->label_show->setPixmap(QPixmap::fromImage(Image));
    ui->label_show->setAlignment(Qt::AlignCenter);
}
void FaceDialog::on_horizontalSlider_wihte_valueChanged(int value)
{
    cv::Mat tempimg = src.clone();
    double scale = 1.3;
    cv::CascadeClassifier cascade = loadCascadeClassifier("./xml/haarcascade_frontalface_alt.xml");//人脸的训练数据
    cv::CascadeClassifier netcascade = loadCascadeClassifier("./xml/haarcascade_eye_tree_eyeglasses.xml");//人眼的训练数据
    if (cascade.empty() || netcascade.empty())
        return;
    detectAndDraw(tempimg, cascade, scale, value);
    if (isDetected == false)
    {
        cout << "enter" << endl;
        cv::Mat dst;

        int value1 = 3, value2 = 1;

        int dx = value1 * 5;    //双边滤波参数之一
        //double fc = value1 * 12.5; //双边滤波参数之一
        double fc = value;
        int p = 50;//透明度
        cv::Mat temp1, temp2, temp3, temp4;

        //对原图层image进行双边滤波，结果存入temp1图层中
        bilateralFilter(tempimg, temp1, dx, fc, fc);

        //将temp1图层减去原图层image，将结果存入temp2图层中
        temp2 = (temp1 - tempimg + 128);

        //高斯模糊
        GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

        //以原图层image为基色，以temp3图层为混合色，将两个图层进行线性光混合得到图层temp4
        temp4 = tempimg + 2 * temp3 - 255;

        //考虑不透明度，修正上一步的结果，得到最终图像dst
        dst = (tempimg*(100 - p) + temp4 * p) / 100;
        dst.copyTo(tempimg);
    }
    img = QCVMat2QImage(tempimg);
    ShowImage();
}

cv::CascadeClassifier FaceDialog::loadCascadeClassifier(const string cascadePath)
{
    cv::CascadeClassifier cascade;
    if (!cascadePath.empty())
    {
        if(!cascade.load(cascadePath))//从指定的文件目录中加载级联分类器
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
        }
    }
    return cascade;
}
std::vector<std::vector<cv::Point2f>> FaceDialog::dectectFace68(QString filename)
{
    std::vector<std::vector<cv::Point2f>>  rets;
    //加载图片路径
    dlib::array2d<rgb_pixel> img;
    QTextCodec *code = QTextCodec::codecForName("gb2312");
    std::string path = code->fromUnicode(filename).data();
    load_image(img, path.c_str());
    //定义人脸检测器
    frontal_face_detector detector = get_frontal_face_detector();
    std::vector<dlib::rectangle> dets = detector(img);

    for (auto var : dets)
    {
        //关键点检测器
        shape_predictor sp;
        deserialize("./xml/shape_predictor_68_face_landmarks.dat") >> sp;
        //定义shape对象保存检测的68个关键点
        full_object_detection shape = sp(img, var);
        //存储文件
        ofstream out("face_detector.txt");
        //读取关键点到容器中
        std::vector<cv::Point2f> points_vec;
        for (int i = 0; i < shape.num_parts(); ++i)
        {
            auto a = shape.part(i);
            out << a.x() << " " << a.y() << " ";
            cv::Point2f ff(a.x(), a.y());
            points_vec.push_back(ff);
        }
        rets.push_back(points_vec);
    }
    return rets;
}

void FaceDialog::detectAndDraw(cv::Mat& img, cv::CascadeClassifier& cascade,  double scale, int val)
{
    std::vector<cv::Rect> faces;
    const static cv::Scalar colors[] = { CV_RGB(0,0,255),CV_RGB(0,128,255),CV_RGB(0,255,255),CV_RGB(0,255,0),CV_RGB(255,128,0),CV_RGB(255,255,0),CV_RGB(255,0,0),CV_RGB(255,0,255) };//用不同的颜色表示不同的人脸
    //将图片缩小，加快检测速度
    cv::Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
    //因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像
    cvtColor(img, gray, CV_BGR2GRAY);
    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);//将尺寸缩小到1/scale,用线性插值
    equalizeHist(smallImg, smallImg);//直方图均衡
    cascade.detectMultiScale(smallImg, //image表示的是要检测的输入图像
        faces,//objects表示检测到的人脸目标序列
        1.1, //caleFactor表示每次图像尺寸减小的比例
        2, //minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
        0 | cv::CASCADE_SCALE_IMAGE ,//minSize为目标的最小尺寸
        cv::Size(30, 30)); //minSize为目标的最大尺寸
    int i = 0;
    //遍历检测的矩形框
    for (std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
    {
        isDetected = true;
        cv::Mat smallImgROI;
        std::vector<cv::Rect> nestedObjects;
        cv::Point center, left, right;
        cv::Scalar color = colors[i % 8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);//还原成原来的大小
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);

        left.x = center.x - radius;
        left.y = cvRound(center.y - radius * 1.3);

        if (left.y < 0)
        {
            left.y = 0;
        }
        right.x = center.x + radius;
        right.y = cvRound(center.y + radius * 1.3);

        if (right.y > img.rows)
        {
            right.y = img.rows;
        }
        /*原理算法
        美肤-磨皮算法
        Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
        */
        //绘画识别的人脸框
        //rectangle(img, left, right, Scalar(255, 0, 0));
        cv::Mat roi = img(cv::Range(left.y, right.y), cv::Range(left.x, right.x));

        cv::Mat dst;
        int value1 = 3, value2 = 1;

        int dx = value1 * 5;    //双边滤波参数之一
        //double fc = value1 * 12.5; //双边滤波参数之一
        double fc = val;//变化值
        int p = 50;//透明度
        cv::Mat temp1, temp2, temp3, temp4;

        //双边滤波    输入图像 输出图像 每像素领域的直径范围颜色空间过滤器的sigma  坐标空间滤波器的sigma
        bilateralFilter(roi, temp1, dx, fc, fc);
        temp2 = (temp1 - roi + 128);
        //高斯模糊
        GaussianBlur(temp2, temp3, cv::Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
        temp4 = roi + 2 * temp3 - 255;
        dst = (roi*(100 - p) + temp4 * p) / 100;
        dst.copyTo(roi);
    }
}

void FaceDialog::on_horizontalSlider_wihte_2_valueChanged(int value)
{
    cv::Mat m = cv::Mat::zeros(src.size(), src.type());
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    m = cv::Scalar(value, value, value);
    cv::addWeighted(src, 1.0, m, 0, value, dst);
    img = QCVMat2QImage(dst);
    ShowImage();
}
void FaceDialog::whiteFace(cv::Mat& matSelfPhoto,int alpha, int beta)
{
    for (int y = 0; y < matSelfPhoto.rows; y++)
    {
        for (int x = 0; x < matSelfPhoto.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                matSelfPhoto.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha*(matSelfPhoto.at<cv::Vec3b>(y, x)[c]) + beta);
            }
        }
    }
}

void FaceDialog::on_pushButton_clicked()
{
    cv::Mat matResult;
    matResult = src.clone();
    int bilateralFilterVal = 30;
    whiteFace(matResult, 1.1, 68);
    GaussianBlur(matResult, matResult, cv::Size(9, 9), 0, 0);
    bilateralFilter(src, matResult, bilateralFilterVal, // 整体磨皮
            bilateralFilterVal * 2, bilateralFilterVal / 2);
    cv::Mat matFinal;

    // 图像增强，使用非锐化掩蔽（Unsharpening Mask）方案。
    cv::GaussianBlur(matResult, matFinal, cv::Size(0, 0), 9);
    cv::addWeighted(matResult, 1.5, matFinal, -0.5, 0, matFinal);
    dst = matFinal.clone();
    img = QCVMat2QImage(dst);
    ShowImage();
}
void FaceDialog::BilinearInsert(cv::Mat &src, cv::Mat &dst, float ux, float uy, int i, int j)
{
    auto Abs = [&](float f) {
        return f > 0 ? f : -f;
    };

    int c = src.channels();
    if (c == 3)
    {
        //存储图像得浮点坐标
        CvPoint2D32f uv;
        CvPoint3D32f f1;
        CvPoint3D32f f2;

        //取整数
        int iu = (int)ux;
        int iv = (int)uy;
        uv.x = iu + 1;
        uv.y = iv + 1;

        //step图象像素行的实际宽度  三个通道进行计算(0 , 1 2  三通道)
        f1.x = ((uchar*)(src.data + src.step*iv))[iu * 3] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*iv))[(iu + 1) * 3] * (uv.x - iu);
        f1.y = ((uchar*)(src.data + src.step*iv))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 1] * (uv.x - iu);
        f1.z = ((uchar*)(src.data + src.step*iv))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 2] * (uv.x - iu);


        f2.x = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3] * (uv.x - iu);
        f2.y = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 1] * (uv.x - iu);
        f2.z = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
            ((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 2] * (uv.x - iu);

        ((uchar*)(dst.data + dst.step*j))[i * 3] = f1.x*(1 - Abs(uv.y - iv)) + f2.x*(Abs(uv.y - iv));  //三个通道进行赋值
        ((uchar*)(dst.data + dst.step*j))[i * 3 + 1] = f1.y*(1 - Abs(uv.y - iv)) + f2.y*(Abs(uv.y - iv));
        ((uchar*)(dst.data + dst.step*j))[i * 3 + 2] = f1.z*(1 - Abs(uv.y - iv)) + f2.z*(Abs(uv.y - iv));

    }
}
cv::Mat FaceDialog::LocalTranslationWarp_Face(cv::Mat &imag, int warpX, int warpY, int endX, int endY, float radius)
{
    cv::Mat dst = imag.clone();
    //平移距离
    float ddradius = radius * radius;
    //计算|m-c|^2
    size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
    //计算 图像的高  宽 通道数量
    int height = imag.rows;
    int width = imag.cols;
    int chan = imag.channels();

    auto Abs = [&](float f) {
        return f > 0 ? f : -f;
    };

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            // # 计算该点是否在形变圆的范围之内
            //# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
            if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
                continue;

            float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
            if (distance < ddradius)
            {
                //# 计算出（i, j）坐标的原坐标
                //# 计算公式中右边平方号里的部分
                float ratio = (ddradius - distance) / (ddradius - distance + mc);
                ratio *= ratio;

                //映射原位置
                float UX = i - ratio * (endX - warpX);
                float UY = j - ratio * (endY - warpY);

                //根据双线性插值得到UX UY的值
                BilinearInsert(imag, dst, UX, UY, i, j);
                //改变当前的值
            }
        }
    }

    return dst;

}
void FaceDialog::on_horizontalSlider_2_valueChanged(int value)
{
    cv::Mat dst = src.clone();
    for (auto points_vec : m_vecFaceData)
    {
        cv::Point2f endPt = points_vec[34];
        for (int i = 3; i < 15; i = i + 2)
        {
            cv::Point2f start_landmark = points_vec[i];
            cv::Point2f end_landmark = points_vec[i + 2];

            float dis = value;
            dst = LocalTranslationWarp_Face(dst, start_landmark.x, start_landmark.y, endPt.x, endPt.y, dis);

        }
    }
    img=QCVMat2QImage(dst);
    ShowImage();

}
void FaceDialog::LocalTranslationWarp_Eye(cv::Mat &img, cv::Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
{
    //平移距离
    float ddradius = radius * radius;
    //计算|m-c|^2
    size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
    //计算 图像的高  宽 通道数量
    int height = img.rows;
    int width = img.cols;
    int chan = img.channels();

    auto Abs = [&](float f) {
        return f > 0 ? f : -f;
    };

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            // # 计算该点是否在形变圆的范围之内
            //# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
            if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
                continue;

            float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
            if (distance < ddradius)
            {
                float rnorm = sqrt(distance) / radius;
                float ratio = 1 - (rnorm - 1)*(rnorm - 1)*0.5;
                //映射原位置
                float UX = warpX + ratio * (i - warpX);
                float UY = warpY + ratio * (j - warpY);

                //根据双线性插值得到UX UY的值
                BilinearInsert(img, dst, UX, UY, i, j);
            }
        }
    }

}
void FaceDialog::on_horizontalSlider_3_valueChanged(int value)
{
    cv::Mat dst = src.clone();
        for (auto points_vec : m_vecFaceData)
        {
            cv::Point2f left_landmark = points_vec[38];
            cv::Point2f	left_landmark_down = points_vec[27];

            cv::Point2f	right_landmark = points_vec[44];
            cv::Point2f	right_landmark_down = points_vec[27];

            cv::Point2f	endPt = points_vec[30];

            //# 计算第4个点到第6个点的距离作为距离
            /*float r_left = sqrt(
                (left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
                (left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
            cout << "左眼距离:" << r_left;*/
            float r_left = value;

            //	# 计算第14个点到第16个点的距离作为距离
            //float	r_right = sqrt(
            //	(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
            //	(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
            //cout << "右眼距离:" << r_right;
            float r_right = value;
            //	# 瘦左
            LocalTranslationWarp_Eye(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
            //	# 瘦右
            LocalTranslationWarp_Eye(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

        }
        img = QCVMat2QImage(dst);
        ShowImage();

}
