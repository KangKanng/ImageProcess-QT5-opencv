#include "concatdialog.h"
#include "ui_concatdialog.h"

ConcatDialog::ConcatDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConcatDialog)
{
    ui->setupUi(this);
}

ConcatDialog::~ConcatDialog()
{
    delete ui;
}

//将cv::Mat转换为QImage
QImage ConcatDialog::QCVMat2QImage(const cv::Mat& mat)
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
QImage ConcatDialog::ImageCenter(QImage qimage, QLabel *qLabel)
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
void ConcatDialog::ShowImage(QLabel *qLabel)
{
    QImage Image=ImageCenter(img,qLabel);
    qLabel->setPixmap(QPixmap::fromImage(Image));
    qLabel->setAlignment(Qt::AlignCenter);
}
void ConcatDialog::on_pushButton_open1_clicked()
{
    filename1 = QFileDialog::getOpenFileName(this, "Open Image", ".", "Image Files(*.png *.jpg *.jpeg *.bmp)");
    if(!filename1.isEmpty())
    {
        QTextCodec *code = QTextCodec::codecForName("gb2312");
        std::string name = code->fromUnicode(filename1).data();
        src1 = cv::imread(name);
        img = QCVMat2QImage(src1);
        ShowImage(ui->label_show1);
    }
}

void ConcatDialog::on_pushButton_open2_clicked()
{
    filename2 = QFileDialog::getOpenFileName(this, "Open Image", ".", "Image Files(*.png *.jpg *.jpeg *.bmp)");
    if(!filename2.isEmpty())
    {
        QTextCodec *code = QTextCodec::codecForName("gb2312");
        std::string name = code->fromUnicode(filename2).data();
        src2 = cv::imread(name);
        img = QCVMat2QImage(src2);
        ShowImage(ui->label_show2);
    }
}

void ConcatDialog::on_pushButton_vertical_clicked()
{
    if(src1.cols>src2.cols)
        cv::resize(src2, src2, cv::Size(src1.cols, src2.rows*src1.cols/src2.cols), 0, 0, 3);
    else if(src1.cols<src2.cols)
        cv::resize(src2, src2, cv::Size(src1.cols, src2.rows*src1.cols/src2.cols), 0, 0, 4);
    vector<cv::Mat>vImgs;
    vImgs.push_back(src1);
    vImgs.push_back(src2);
    vconcat(vImgs, dst);
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

void ConcatDialog::on_pushButton_horizomal_clicked()
{
    if(src1.rows>src2.rows)
        cv::resize(src2, src2, cv::Size(src2.cols*src1.rows/src2.rows, src1.rows), 0, 0, 3);
    else if(src1.rows<src2.rows)
        cv::resize(src2, src2, cv::Size(src2.cols*src1.rows/src2.rows, src1.rows), 0, 0, 4);
    vector<cv::Mat>vImgs;
    vImgs.push_back(src1);
    vImgs.push_back(src2);
    hconcat(vImgs, dst);
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

void ConcatDialog::on_pushButton_model_clicked()
{
    double start = cv::getTickCount();
    cv::Mat grayL, grayR;
    cvtColor(src1, grayL, cv::COLOR_BGR2GRAY);
    cvtColor(src2, grayR, cv::COLOR_BGR2GRAY);

    cv::Rect rectCut = cv::Rect(372, 122, 128, 360);
    cv::Rect rectMatched = cv::Rect(0, 0, src2.cols / 2, src2.rows);
    cv::Mat imgTemp = grayL(cv::Rect(rectCut));
    cv::Mat imgMatched = grayR(cv::Rect(rectMatched));

      int width = imgMatched.cols - imgTemp.cols + 1;
      int height = imgMatched.rows - imgTemp.rows + 1;
      cv::Mat matchResult(height, width, CV_32FC1);
      matchTemplate(imgMatched, imgTemp, matchResult, cv::TM_CCORR_NORMED);
      normalize(matchResult, matchResult, 0, 1, cv::NORM_MINMAX, -1);  //归一化到0--1范围

      double minValue, maxValue;
      cv::Point minLoc, maxLoc;
      minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);

      cv::Mat dstImg(src1.rows, src2.cols + rectCut.x - maxLoc.x, CV_8UC3, cv::Scalar::all(0));
      cv::Mat roiLeft = dstImg(cv::Rect(0, 0, src1.cols, src1.rows));
      src1.copyTo(roiLeft);

      cv::Mat debugImg = src2.clone();
      rectangle(debugImg, cv::Rect(maxLoc.x, maxLoc.y, imgTemp.cols, imgTemp.rows), cv::Scalar(0, 255, 0), 2, 8);
      //imwrite("match.jpg", debugImg);

      cv::Mat roiMatched = src2(cv::Rect(maxLoc.x, maxLoc.y - rectCut.y, src2.cols - maxLoc.x, src2.rows - 1 - (maxLoc.y - rectCut.y)));
      cv::Mat roiRight = dstImg(cv::Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));

      roiMatched.copyTo(roiRight);

      double end = cv::getTickCount();
      double useTime = (end - start) / cv::getTickFrequency();
      cout << "use-time : " << useTime << "s" << endl;
      dst = dstImg.clone();
      img = QCVMat2QImage(dst);
      ShowImage(ui->label_show);
}

void ConcatDialog::on_pushButton_SIFT_clicked()
{
    ImageOverlap0(src1,src2);
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

bool ConcatDialog::ImageOverlap0(cv::Mat &img1, cv::Mat &img2)
{
  cv::Mat g1(img1, cv::Rect(0, 0, img1.cols, img1.rows));  // init roi
  cv::Mat g2(img2, cv::Rect(0, 0, img2.cols, img2.rows));

  cvtColor(g1, g1, cv::COLOR_BGR2GRAY);
  cvtColor(g2, g2, cv::COLOR_BGR2GRAY);

  vector<cv::KeyPoint> keypoints_roi, keypoints_img;  /* keypoints found using SIFT */
  cv::Mat descriptor_roi, descriptor_img;                           /* Descriptors for SIFT */
  cv::FlannBasedMatcher matcher;                                   /* FLANN based matcher to match keypoints */

  vector<cv::DMatch> matches, good_matches;
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
  int i, dist = 80;

  sift->detectAndCompute(g1, cv::Mat(), keypoints_roi, descriptor_roi);      /* get keypoints of ROI image */
  sift->detectAndCompute(g2, cv::Mat(), keypoints_img, descriptor_img);         /* get keypoints of the image */
  matcher.match(descriptor_roi, descriptor_img, matches);  //实现描述符之间的匹配

  double max_dist = 0; double min_dist = 5000;
  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptor_roi.rows; i++)
  {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  // 特征点筛选
  for (i = 0; i < descriptor_roi.rows; i++)
  {
    if (matches[i].distance < 3 * min_dist)
    {
      good_matches.push_back(matches[i]);
    }
  }

  printf("%ld no. of matched keypoints in right image\n", good_matches.size());
  /* Draw matched keypoints */

  cv::Mat img_matches;
  //绘制匹配
  drawMatches(img1, keypoints_roi, img2, keypoints_img,
  good_matches, img_matches, cv::Scalar::all(-1),
    cv::Scalar::all(-1), vector<char>(),
    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //imshow("matches", img_matches);

  vector<cv::Point2f> keypoints1, keypoints2;
  for (i = 0; i < good_matches.size(); i++)
  {
    keypoints1.push_back(keypoints_img[good_matches[i].trainIdx].pt);
    keypoints2.push_back(keypoints_roi[good_matches[i].queryIdx].pt);
  }
  //计算单应矩阵(仿射变换矩阵)
  cv::Mat H = findHomography(keypoints1, keypoints2, cv::RANSAC);
  cv::Mat H2 = findHomography(keypoints2, keypoints1, cv::RANSAC);


  cv::Mat stitchedImage;  //定义仿射变换后的图像(也是拼接结果图像)
  cv::Mat stitchedImage2;  //定义仿射变换后的图像(也是拼接结果图像)
  int mRows = img2.rows;
  if (img1.rows > img2.rows)
  {
    mRows = img1.rows;
  }

  int count = 0;
  for (int i = 0; i < keypoints2.size(); i++)
  {
    if (keypoints2[i].x >= img2.cols / 2)
      count++;
  }
  //判断匹配点位置来决定图片是左还是右
  if (count / float(keypoints2.size()) >= 0.5)  //待拼接img2图像在右边
  {
    cout << "img1 should be left" << endl;
    vector<cv::Point2f>corners(4);
    vector<cv::Point2f>corners2(4);
    corners[0] = cv::Point(0, 0);
    corners[1] = cv::Point(0, img2.rows);
    corners[2] = cv::Point(img2.cols, img2.rows);
    corners[3] = cv::Point(img2.cols, 0);
    stitchedImage = cv::Mat::zeros(img2.cols + img1.cols, mRows, CV_8UC3);
    warpPerspective(img2, stitchedImage, H, cv::Size(img2.cols + img1.cols, mRows));

    perspectiveTransform(corners, corners2, H);
    /*
    circle(stitchedImage, corners2[0], 5, Scalar(0, 255, 0), 2, 8);
    circle(stitchedImage, corners2[1], 5, Scalar(0, 255, 255), 2, 8);
    circle(stitchedImage, corners2[2], 5, Scalar(0, 255, 0), 2, 8);
    circle(stitchedImage, corners2[3], 5, Scalar(0, 255, 0), 2, 8); */
    cout << corners2[0].x << ", " << corners2[0].y << endl;
    cout << corners2[1].x << ", " << corners2[1].y << endl;
    //imshow("temp", stitchedImage);
    //imwrite("temp.jpg", stitchedImage);

    cv::Mat half(stitchedImage, cv::Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);
    //imshow("result", stitchedImage);
    dst = stitchedImage;
  }
  else  //待拼接图像img2在左边
  {
    cout << "img2 should be left" << endl;
    stitchedImage = cv::Mat::zeros(img2.cols + img1.cols, mRows, CV_8UC3);
    warpPerspective(img1, stitchedImage, H2, cv::Size(img1.cols + img2.cols, mRows));
    //imshow("temp", stitchedImage);

    //计算仿射变换后的四个端点
    vector<cv::Point2f>corners(4);
    vector<cv::Point2f>corners2(4);
    corners[0] = cv::Point(0, 0);
    corners[1] = cv::Point(0, img1.rows);
    corners[2] = cv::Point(img1.cols, img1.rows);
    corners[3] = cv::Point(img1.cols, 0);

    perspectiveTransform(corners, corners2, H2);  //仿射变换对应端点
    /*
    circle(stitchedImage, corners2[0], 5, Scalar(0, 255, 0), 2, 8);
    circle(stitchedImage, corners2[1], 5, Scalar(0, 255, 255), 2, 8);
    circle(stitchedImage, corners2[2], 5, Scalar(0, 255, 0), 2, 8);
    circle(stitchedImage, corners2[3], 5, Scalar(0, 255, 0), 2, 8); */
    cout << corners2[0].x << ", " << corners2[0].y << endl;
    cout << corners2[1].x << ", " << corners2[1].y << endl;

    cv::Mat half(stitchedImage, cv::Rect(0, 0, img2.cols, img2.rows));
    img2.copyTo(half);
    //imshow("result", stitchedImage);
    dst = stitchedImage;

  }
  //imwrite("result.bmp", stitchedImage);
  return true;
}


void ConcatDialog::on_pushButton_SURF_clicked()
{
    vector<cv::Mat> imgs;
    imgs.push_back(src1);
    imgs.push_back(src2);
    cv::Stitcher stitcher = cv::Stitcher::createDefault(true);
    stitcher.stitch(imgs, dst);
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

void ConcatDialog::on_pushButton_bend_clicked()
{
    vector<cv::Mat> imgs;
    imgs.push_back(src1);
    imgs.push_back(src2);
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    auto blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND);
    stitcher->setBlender(blender);
    auto plane_warper = cv::makePtr<cv::PlaneWarper>();
    stitcher->setWarper(plane_warper);
    cv::Stitcher::Status status = stitcher->stitch(imgs, dst);
    if (status != cv::Stitcher::OK)
    {
        cout << "不能拼接 " << int(status) << endl;
        QMessageBox::warning(this,"错误","无法拼接，请使用两张有重合区域的图片！");
        return;
    }
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

void ConcatDialog::on_pushButton_fisheye_clicked()
{
    vector<cv::Mat> imgs;
    imgs.push_back(src1);
    imgs.push_back(src2);
    cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    auto blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND);
    stitcher->setBlender(blender);
    auto fisheye_warper = cv::makePtr<cv::FisheyeWarper>();
    stitcher->setWarper(fisheye_warper);
    cv::Stitcher::Status status = stitcher->stitch(imgs, dst);
    if (status != cv::Stitcher::OK)
    {
        cout << "不能拼接 " << int(status) << endl;
        QMessageBox::warning(this,"错误","无法拼接，请使用两张有重合区域的图片！");
        return;
    }
    img = QCVMat2QImage(dst);
    ShowImage(ui->label_show);
}

void ConcatDialog::on_buttonBox_accepted()
{
    sendMat(dst);
}
