#-------------------------------------------------
#
# Project created by QtCreator 2022-06-02T16:31:03
#
#-------------------------------------------------

QT       += core gui
QT       += multimedia
QT       += multimediawidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = opencv_bigwork
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    imagecropperlabel.cpp \
    noisedialog.cpp \
    imagecropper.cpp \
    filterdialog.cpp \
    concatdialog.cpp \
    facedialog.cpp

HEADERS += \
        mainwindow.h \
    imagecropperdialog.h \
    imagecropperlabel.h \
    noisedialog.h \
    imagecropper.h \
    filterdialog.h \
    concatdialog.h \
    facedialog.h

FORMS += \
        mainwindow.ui \
    noisedialog.ui \
    filterdialog.ui \
    concatdialog.ui \
    facedialog.ui

INCLUDEPATH += \
                E:\Path\opencv3416\opencv\build\install\include \
                E:\Path\dlib-19.24\include
LIBS += \
        E:\Path\opencv3416\opencv\build\lib\libopencv_*.a \
        E:\Path\dlib-19.24\lib\libdlib.a


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    icons.qrc
