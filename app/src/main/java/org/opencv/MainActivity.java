package org.opencv;

import android.Manifest;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.TimingLogger;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = "TemplateMatching";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_13, this, new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1000);
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //setContentView(R.layout.activity_main);
        new Thread(new Runnable() {
            @Override
            public void run() {
                Mat img = Highgui.imread("/sdcard/1.png");
                Mat templ = Highgui.imread("/sdcard/2.png");
                Mat templ1 = Highgui.imread("/sdcard/3.png");
                Log.d(LOG_TAG, tmPyr(img.clone(), templ, Imgproc.TM_CCOEFF_NORMED).toString());
                Log.d(LOG_TAG, tmPyr(img.clone(), templ1, Imgproc.TM_CCOEFF_NORMED).toString());
            }
        }).start();

    }

    private Mat scale(Mat mat, float f) {
        int r = (int) (mat.rows() * f);
        int c = (int) (mat.cols() * f);
        Mat m = new Mat(r, c, mat.type());
        Imgproc.resize(mat, m, m.size());
        return m;
    }

    public static Point tmPyr(Mat img, Mat template, int match_method) {
        TimingLogger logger = new TimingLogger(LOG_TAG, "tmPyr");
        int maxLevel = 5;
        List<Mat> imgPyr = buildPyramid(img, maxLevel);
        List<Mat> tmplPry = buildPyramid(template, maxLevel);
        Point p = null;
        Mat result = null;
        for (int level = maxLevel; level >= 0; level--) {
            Mat src = imgPyr.get(level);
            Mat tmpl = tmplPry.get(level);
            if (level == maxLevel) {
                result = matchTemplate(src, tmpl, match_method);
                p = getBestMatch(result, match_method);
            } else {
                int x = (int) (p.x * 2 - tmpl.rows() / 4);
                x = Math.max(0, x);
                int y = (int) (p.y * 2 - tmpl.cols() / 4);
                y = Math.max(0, y);
                int w = (int) (tmpl.rows() * 1.5);
                int h = (int) (tmpl.cols() * 1.5);
                if (x + w >= src.cols()) {
                    w = src.cols() - x - 1;
                }
                if (y + h >= src.rows()) {
                    h = src.rows() - y - 1;
                }
                Rect r = new Rect(x, y, w, h);
                Log.d(LOG_TAG, "r: " + r + ", src: " + src + ", tmpl: " + tmpl);
                result = matchTemplate(new Mat(src, r), tmpl, match_method);
                p = getBestMatch(result, match_method);
                p.x += r.x;
                p.y += r.y;
            }
            //Imgproc.threshold(result, result, 0.94, 1, Imgproc.THRESH_TOZERO);
            logger.addSplit("level:" + level);
            Log.d(LOG_TAG, "level: " + level + " point:" + p);
        }
        logger.dumpToLog();
        return p;
    }
    /*
    Mat mask = new Mat(result.rows() * 2, result.cols() * 2, CvType.CV_8UC1);
                Imgproc.pyrUp(result, mask);
                Mat mask8u = new Mat();
                mask.convertTo(mask8u, CvType.CV_8UC1);
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(mask8u, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
                for (MatOfPoint contour : contours) {
                    Rect r = Imgproc.boundingRect(contour);
                    result = matchTemplate(new Mat(src, r), tmpl, match_method);
                }
     */

    public static List<Mat> buildPyramid(Mat mat, int maxLevel) {
        List<Mat> pyramid = new ArrayList<>();
        pyramid.add(mat);
        for (int i = 0; i < maxLevel; i++) {
            Mat m = new Mat((mat.rows() + 1) / 2, (mat.cols() + 1) / 2, mat.type());
            Imgproc.pyrDown(mat, m);
            pyramid.add(m);
            mat = m;
        }
        return pyramid;
    }

    public static Mat matchTemplate(Mat img, Mat temp, int match_method) {
        int result_cols = img.cols() - temp.cols() + 1;
        int result_rows = img.rows() - temp.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);
        Imgproc.matchTemplate(img, temp, result, match_method);
        return result;
    }

    public static Point getBestMatch(Mat tmResult, int match_method) {
        Core.normalize(tmResult, tmResult, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        Core.MinMaxLocResult mmr = Core.minMaxLoc(tmResult);
        if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
            return mmr.minLoc;
        } else {
            return mmr.maxLoc;
        }
    }

    public static Point tm(Mat img, Mat template, int match_method) {
        TimingLogger logger = new TimingLogger(LOG_TAG, "tm");
        int result_cols = img.cols() - template.cols() + 1;
        int result_rows = img.rows() - template.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);
        Imgproc.matchTemplate(img, template, result, match_method);
        logger.addSplit("tm");
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        logger.addSplit("normalize");

        // / Localizing the best match with minMaxLoc
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        Point matchLoc;
        if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
            matchLoc = mmr.minLoc;
        } else {
            matchLoc = mmr.maxLoc;
        }
        logger.dumpToLog();
        return matchLoc;
    }

}
