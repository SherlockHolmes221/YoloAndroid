package com.example.yolo

import android.content.res.AssetManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import com.example.yolo.Tensorflow.YoloDetection
import kotlinx.android.synthetic.main.activity_main.*
import android.util.Log
import com.example.yolo.Tensorflow.Detector
import com.example.yolo.bean.BBox
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.lang.Exception
import android.graphics.*


class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    private val MODEL_FILE = "file:///android_asset/model/yolov3_coco.pb"
    private val MODEL_CLASS = "classes.txt"
    private val CLASS_NUM = 80
    private val IMAGE_NAME = "road.jpeg"
    private val INPUT_SIZE = 416
    private val INPUT_NAME = "input/input_data:0"
    private val OUTPUT_NAME = arrayOf("pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2")
    private val OUTPUTSIZE = arrayOf(52*52*5*80,26*26*5*80,13*13*5*80)

    private lateinit var detector : Detector
    private lateinit var originImage : Bitmap
    private lateinit var bitmap : Bitmap
    private var classArray : MutableList<String> = arrayListOf()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        detector = getModel()

        originImage = getImageFromAsset()
        Log.i(TAG, "width:" + originImage.width+ " height:" + originImage.height)
        //main_iv.setImageBitmap(originImage)

        getClassesFromAsset()

        bitmap = detector.getScaleBitmap(originImage, INPUT_SIZE)
        main_iv.setImageBitmap(bitmap)

        GlobalScope.launch(Dispatchers.Main) {
            refreshBoxes(GlobalScope.async {
                detect()
            }.await())
        }
    }

    private fun refreshBoxes(await: MutableList<BBox>) {
        await.sortByDescending {it.score}
        Log.i(TAG, "refreshBoxes")
        Log.i("size", await.size.toString())

        val newBitmap = originImage.copy(Bitmap.Config.ARGB_8888, true)
        Log.i(TAG, "newBitmap: "+newBitmap.width)
        Log.i(TAG, "newBitmap: "+newBitmap.height)

        for(bbox in await){
            val xmin = bbox.xmin
            val xmax = bbox.xmax
            val ymin = bbox.ymin
            val ymax = bbox.ymax
            val classid= bbox.classId

            Log.i(TAG, "xmin: "+xmin)
            Log.i(TAG, "ymin: "+ymin)
            Log.i(TAG, "xmax: "+xmax)
            Log.i(TAG, "ymax: "+ymax)
            Log.i(TAG, "classid: "+classid)
            // get class name
            val className = classArray.get(classid)

            //draw bbox
            val canvas = Canvas(newBitmap)
            val paint = Paint()
            val rect = Rect(xmin.toInt(), ymin.toInt(), xmax.toInt(), ymax.toInt())
            paint.setColor(Color.RED)
            paint.setStyle(Paint.Style.STROKE)
            paint.setStrokeWidth(5.0f)
            canvas.drawRect(rect, paint)
            canvas.drawBitmap(newBitmap, rect, rect, paint)
//            canvas.drawText(className+bbox.score, xmin,ymin, paint)
        }
        main_iv.setImageBitmap(newBitmap)

    }

    private fun detect(): MutableList<BBox> {
        Log.i("detect_image", "detect")
        var list =  detector.detect(bitmap)

        //todo
        return list
    }


    fun getImageFromAsset():Bitmap{
        val assets: AssetManager = assets
        val resource = assets.open(IMAGE_NAME)
        return BitmapFactory.decodeStream(resource)
    }

    fun getClassesFromAsset(){
        val assets: AssetManager = assets
        val inputStream = assets.open(MODEL_CLASS)

        var inputStreamReader :InputStreamReader ?= null
        try {
            inputStreamReader=InputStreamReader(inputStream, "UTF-8")
        } catch (e:Exception ) {
            e.printStackTrace()
        }
        val reader = BufferedReader(inputStreamReader)

        try {
            var name = reader.readLine()
            while (name != null) {
                val temp = name
                classArray.add(temp)
                name = reader.readLine()
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        Log.i(TAG, classArray.get(79))
    }

    fun getModel() : YoloDetection{
        return YoloDetection.create(this.assets, MODEL_FILE, INPUT_SIZE,OUTPUTSIZE,INPUT_NAME,OUTPUT_NAME,this)
    }

    override fun onDestroy() {
        detector.close()
        super.onDestroy()
    }

    fun showToast(s :String){
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show()
    }

    fun getIou(){

    }

}
