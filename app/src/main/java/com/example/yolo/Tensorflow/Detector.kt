package com.example.yolo.Tensorflow

import android.graphics.Bitmap

interface Detector {
    override fun toString(): String
    fun close()
    fun detect(bitmap: Bitmap)
}