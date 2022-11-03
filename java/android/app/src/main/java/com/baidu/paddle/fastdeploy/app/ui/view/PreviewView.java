package com.baidu.paddle.fastdeploy.app.ui.view;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.AttributeSet;
import android.view.Surface;
import android.view.TextureView;

import com.baidu.paddle.fastdeploy.app.ui.camera.CameraListener;
import com.baidu.paddle.fastdeploy.app.ui.camera.CameraProxy1;
import com.baidu.paddle.fastdeploy.app.ui.camera.ICameraProxy;

/**
 * Created by ruanshimin on 2018/5/3.
 */

public class PreviewView extends TextureView {
    ICameraProxy mCameraProxy;
    private int layoutWidth;
    private int layoutHeight;
    private int actualHeight;
    private int cropHeight;

    public PreviewView(Context context) {
        super(context);
    }

    public PreviewView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public PreviewView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    /**
     * 手动调用开始预览
     */
    public void start() {
        if (mCameraProxy != null) {
            mCameraProxy.startPreview();
        }
    }

    /**
     * 释放摄像头
     */
    public void destory() {
        if (mCameraProxy != null) {
            mCameraProxy.closeCamera();
        }
    }

    /**
     * 手动停止
     */
    public void stopPreview() {
        mCameraProxy.stopPreview();
    }

    /**
     * 设置实际可见layout的长宽
     */
    public void setLayoutSize(int width, int height) {
        layoutWidth = width;
        layoutHeight = height;
    }

    public void takePicture(final CameraListener.TakePictureListener listener) {
        // 裁剪图片
        mCameraProxy.takePicture(new CameraListener.TakePictureListener() {
            @Override
            public void onTakenPicture(Bitmap bitmap) {
                Bitmap cropBitmap = Bitmap.createBitmap(bitmap, 0, 0,
                        bitmap.getWidth(), cropHeight);
                listener.onTakenPicture(cropBitmap);
            }
        });
    }

    /**
     * 切换摄像头面
     */
    public void switchSide() {
        mCameraProxy.switchSide();
    }

    public int getActualHeight() {
        return actualHeight;
    }

    private void setDisplayDegree() {
        int rotation = ((Activity) this.getContext()).getWindowManager().getDefaultDisplay()
                .getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
            default:
                degrees = 0;
        }
        mCameraProxy.setDisplayRotation(degrees);
    }

    private void refreshCropHeight() {
        int[] size = mCameraProxy.getPreviewSize();
        actualHeight = (int) (((float) size[1] / size[0]) * layoutWidth);
        cropHeight = (int) (((float) layoutHeight / layoutWidth) * size[0]);
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        if (mCameraProxy == null) {
            mCameraProxy = new CameraProxy1(layoutWidth, layoutHeight);
            setDisplayDegree();
            setSurfaceTextureListener(mCameraProxy);
            mCameraProxy.openCamera();
            refreshCropHeight();
            mCameraProxy.setEventListener(new CameraListener.CommonListener() {
                @Override
                public void onSurfaceReady() {
                    mCameraProxy.startPreview();
                }

                @Override
                public void onSwitchCamera() {
                    refreshCropHeight();
                }
            });
        }
    }
}
