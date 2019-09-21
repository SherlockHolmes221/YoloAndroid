package com.example.yolo.Tensorflow

import android.graphics.Bitmap
import com.example.yolo.bean.BBox

interface Detector {
    override fun toString(): String
    fun close()
    fun detect(bitmap: Bitmap):MutableList<BBox>
    fun getScaleBitmap(bitmap: Bitmap, size: Int): Bitmap
}