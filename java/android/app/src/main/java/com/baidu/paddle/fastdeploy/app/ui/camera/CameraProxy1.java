package com.baidu.paddle.fastdeploy.app.ui.camera;

import static android.hardware.Camera.getCameraInfo;
import static android.hardware.Camera.getNumberOfCameras;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.util.Log;

import com.baidu.paddle.fastdeploy.app.ui.util.ImageUtil;
import com.baidu.paddle.fastdeploy.app.ui.util.UiLog;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Created by ruanshimin on 2018/4/16.
 */

public class CameraProxy1 implements ICameraProxy {

    private static final int STATUS_INIT = 0;

    private static final int STATUS_OPENED = 1;

    private static final int STATUS_PREVIEWING = 2;

    private static final int TEXTURE_STATUS_INITED = 0;

    private static final int TEXTURE_STATUS_READY = 1;

    private SurfaceTexture mSurfaceTexture;

    private boolean isBack = true;

    private Camera mCamera;

    private int status;
    private int textureStatus;

    private boolean hasDestoryUnexcepted = false;

    private int layoutWidth;
    private int layoutHeight;

    private int[] previewSize = new int[2];

    private CameraListener.CommonListener mCameraListener;

    private int minPreviewCandidateWidth = 0;

    private int previewRotation = 90;
    private int displayRotation = 0;

    private int previewCaptureRotation = 0;

    private Camera.Parameters cameraParameters;

    public CameraProxy1(int width, int height) {
        layoutWidth = width;
        layoutHeight = height;
        status = STATUS_INIT;
        textureStatus = TEXTURE_STATUS_INITED;
    }

    public int[] getPreviewSize() {
        return previewSize;
    }

    private Camera.Size getPreviewSize(List<Camera.Size> list) {
        ArrayList validSizeList = new ArrayList<Camera.Size>();
        float ratio = (float) layoutHeight / layoutWidth;
        for (Camera.Size size : list) {
            if ((float) size.width / size.height >= ratio && size.width > minPreviewCandidateWidth) {
                validSizeList.add(size);
            }
        }

        // 没有符合条件的，直接返回第一个吧
        if (validSizeList.size() == 0) {
            UiLog.info("no valid preview size");
            return list.get(0);
        }

        Camera.Size size = (Camera.Size) Collections.min(validSizeList, new Comparator<Camera.Size>() {
            @Override
            public int compare(Camera.Size s1, Camera.Size s2) {
                return s1.width - s2.width;
            }
        });

        return (Camera.Size) validSizeList.get(1);
        // return size;
    }


    public Camera open(boolean isOpenBack) {
        int numberOfCameras = getNumberOfCameras();
        UiLog.info("cameraNumber is " + numberOfCameras);
        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            getCameraInfo(i, cameraInfo);
            if (isOpenBack) {
                if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                    caculatePreviewRotation(cameraInfo);
                    return Camera.open(i);
                }
            } else {
                if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    caculatePreviewRotation(cameraInfo);
                    return Camera.open(i);
                }
            }

        }
        // 兼容只有前置摄像头的开发板
        return Camera.open(0);
    }

    public void setDisplayRotation(int degree) {
        displayRotation = degree;
    }

    private void caculatePreviewRotation(Camera.CameraInfo info) {
        int degree;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            degree = (info.orientation + displayRotation) % 360;
            degree = (360 - degree) % 360;  // compensate the mirror
            previewCaptureRotation = (360 - degree) % 360;
        } else {  // back-facing
            degree = (info.orientation - displayRotation + 360) % 360;
            previewCaptureRotation = (360 + degree) % 360;
        }

        previewRotation = degree;
    }

    @Override
    public void startPreview() {
        // MIUI上会在activity stop时destory关闭相机，触发textureview的destory方法
        if (hasDestoryUnexcepted) {
            openCamera();
        }
        // 如果已经TextureAvailable，设置texture并且预览,(必须先调用了openCamera)
        if (textureStatus == TEXTURE_STATUS_READY) {
            try {
                mCamera.setPreviewTexture(mSurfaceTexture);
            } catch (IOException e) {
                e.printStackTrace();
            }

            mCamera.startPreview();
            mCamera.setPreviewCallback(mPreviewCallback);
            status = STATUS_PREVIEWING;
            return;
        }
    }

    public void resumePreview() {
        mCamera.startPreview();
    }

    private byte[] currentFrameData;

    private Camera.PreviewCallback mPreviewCallback = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            // 在某些机型和某项项目中，某些帧的data的数据不符合nv21的格式，需要过滤，否则后续处理会导致crash
            if (data.length != cameraParameters.getPreviewSize().width
                    * cameraParameters.getPreviewSize().height * 1.5) {
                return;
            }
            currentFrameData = data;
        }
    };

    public void openCamera() {
        mCamera = open(isBack);
        cameraParameters = mCamera.getParameters();
        Camera.Size previewSize = cameraParameters.getPreviewSize();
        List<Camera.Size> supportedPreviewSizes = mCamera.getParameters().getSupportedPreviewSizes();
        Camera.Size size = getPreviewSize(supportedPreviewSizes);
        if (previewRotation == 90 || previewRotation == 270) {
            this.previewSize[0] = size.height;
            this.previewSize[1] = size.width;
        } else {
            this.previewSize[0] = size.width;
            this.previewSize[1] = size.height;
        }
        previewSize.height = size.height;
        if (isBack && cameraParameters.getSupportedFocusModes().contains(
                Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
            cameraParameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
        }
        cameraParameters.setPreviewSize(size.width, size.height);
        mCamera.setDisplayOrientation(previewRotation);
        mCamera.setParameters(cameraParameters);
        status = STATUS_OPENED;
        if (mCameraListener != null) {
            mCameraListener.onSwitchCamera();
        }
    }

    public void closeCamera() {
        stopPreview();
        mCamera.setPreviewCallback(null);
        mCamera.release();
        mCamera = null;
        status = STATUS_INIT;
    }

    @Override
    public void stopPreview() {

        mCamera.stopPreview();
        mCamera.setPreviewCallback(null);
        status = STATUS_OPENED;
        Log.e("literrr", "stoped");
    }

    @Override
    public void switchSide() {
        isBack = !isBack;
        stopPreview();
        closeCamera();
        openCamera();
        startPreview();
    }

    private Bitmap convertPreviewDataToBitmap(byte[] data) {
        Camera.Size size = cameraParameters.getPreviewSize();
        YuvImage img = new YuvImage(data, ImageFormat.NV21, size.width, size.height, null);
        ByteArrayOutputStream os = null;
        os = new ByteArrayOutputStream(data.length);
        img.compressToJpeg(new Rect(0, 0, size.width, size.height), 80, os);
        byte[] jpeg = os.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        Bitmap bitmapRotate;
        if (isBack) {
            bitmapRotate = ImageUtil.createRotateBitmap(bitmap, previewCaptureRotation);
        } else {
            bitmapRotate = ImageUtil.createMirrorRotateBitmap(bitmap, previewCaptureRotation);
        }
        try {
            os.close();
        } catch (IOException e) {
            UiLog.info("convertPreviewDataToBitmap close bitmap");
        }
        return bitmapRotate;
    }

    @Override
    public void takePicture(CameraListener.TakePictureListener listener) {
        if (currentFrameData != null) {
            Bitmap bitmap = convertPreviewDataToBitmap(currentFrameData);
            listener.onTakenPicture(bitmap);
            UiLog.info("convert bitmap success");
        }
    }

    @Override
    public void setEventListener(CameraListener.CommonListener listener) {
        mCameraListener = listener;
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        mSurfaceTexture = surface;
        textureStatus = TEXTURE_STATUS_READY;
        mCameraListener.onSurfaceReady();
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        hasDestoryUnexcepted = true;
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {

    }
}
