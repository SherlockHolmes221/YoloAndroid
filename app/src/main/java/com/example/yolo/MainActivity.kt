package com.example.yolo

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import com.example.yolo.Tensorflow.YoloDetection
import kotlinx.android.synthetic.main.activity_main.*
import android.graphics.BitmapFactory
import android.util.Log
import com.example.yolo.Tensorflow.Detector


class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    private val MODEL_FILE = "file:///android_asset/model/yolov3_coco.pb"
    private val IMAGE_NAME = "road.jpeg"
    private val INPUT_SIZE = 416
    private val INPUT_NAME = "input/input_data:0"
    private val OUTPUT_NAME = arrayOf("pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2")
    private val outputSizes = arrayOf(52*52*5*80,26*26*5*80,13*13*5*80)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val detector = getModel()

        val image = getImageFromAsset(IMAGE_NAME)
        Log.i(TAG, "width:" + image.width+ " height:" + image.height)

        detector.detect(image)
    }

    fun getImageFromAsset(imagePath:String):Bitmap{
        val assets: AssetManager = assets
        val resource = assets.open(imagePath)
        return BitmapFactory.decodeStream(resource)
    }

    fun showToast(s :String){
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show()
    }

    fun getModel() : YoloDetection{
        return YoloDetection.create(this.assets, MODEL_FILE, INPUT_SIZE,outputSizes,INPUT_NAME,OUTPUT_NAME)
    }

}
