package com.example.yolo.Tensorflow

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Trace
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.set
import com.example.yolo.bean.BBox
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Exception
import java.nio.ByteBuffer
import java.util.*
import javax.security.auth.login.LoginException


class YoloDetection : Detector{

    private val TAG = "YoloDetection"
    private lateinit var context: Context
    private lateinit var modelFileName:String
    private lateinit var input_name:String
    private lateinit var output_names:Array<String>
    private lateinit var intValues: IntArray
    private var inputSize: Int = 0
    private var floatValues: FloatArray? = null
    private val logStats = false
    private var output1: FloatArray? = null
    private var output2: FloatArray? = null
    private var output3: FloatArray? = null
    private var origin_width = 0
    private var origin_height = 0
    private var scale = 0.0f


    private var inferenceInterface: TensorFlowInferenceInterface? = null

    companion object {
        fun create(assetManager: AssetManager, fileName : String,
                   inputSize:Int, outputSizes: Array<Int>,
                   inputName:String, outputNames: Array<String>,
                   context: Context): YoloDetection{
            var detection = YoloDetection()
            detection.context = context
            detection.modelFileName = fileName

            detection.inferenceInterface = TensorFlowInferenceInterface(assetManager,fileName)
            detection.input_name = inputName
            detection.output_names = outputNames
            detection.inputSize = inputSize

            detection.intValues = IntArray(inputSize * inputSize)
            detection.floatValues = FloatArray(inputSize * inputSize * 3)

            detection.output1 =FloatArray(outputSizes[0])
            detection.output2 =FloatArray(outputSizes[1])
            detection.output3 =FloatArray(outputSizes[2])

            val operation1 = detection.inferenceInterface!!.graphOperation(outputNames[0])
            val operation2 = detection.inferenceInterface!!.graphOperation(outputNames[1])
            val operation3 = detection.inferenceInterface!!.graphOperation(outputNames[2])

            Log.i("YoloDetection", ""+operation1.toString())
            Log.i("YoloDetection", ""+operation2.toString())
            Log.i("YoloDetection", ""+operation3.toString())
            return detection
        }
    }

    override fun detect(bitmap: Bitmap) : MutableList<BBox>{

        Log.i(TAG, ""+bitmap.width+" "+bitmap.height)

        Trace.beginSection("detectImage")

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0,
            bitmap.width, bitmap.height)

        for (i in 0 until intValues.size) {
            val value  = intValues[i]
            //Log.i(TAG, value.toString())
            if(value == 0){
                floatValues!![i * 3 + 0]= 0.50196078f
                floatValues!![i * 3 + 1]= 0.50196078f
                floatValues!![i * 3 + 2]= 0.50196078f
                continue
            }

            floatValues!![i * 3 + 0] = (((value shr 16) and 0xFF) / (255.0)).toFloat()
           // Log.i(TAG, floatValues!![i * 3 + 0].toString())
            floatValues!![i * 3 + 1] = (((value shr 8) and 0xFF) /255.0).toFloat()
           // Log.i(TAG, floatValues!![i * 3 + 1].toString())
            floatValues!![i * 3 + 2] = (((value and 0xFF)) / (255.0)).toFloat()
           // Log.i(TAG, floatValues!![i * 3 + 2].toString())
        }
        Trace.endSection()

        Trace.beginSection("feed")
        inferenceInterface!!.feed(input_name, floatValues, 1,inputSize.toLong(),inputSize.toLong(),3)
        Trace.endSection()

        Trace.beginSection("run")
        inferenceInterface!!.run(output_names, logStats)
        Trace.endSection()

        Trace.beginSection("fetch")
        inferenceInterface!!.fetch(output_names[0], output1)
        inferenceInterface!!.fetch(output_names[1], output2)
        inferenceInterface!!.fetch(output_names[2], output3)
        Trace.endSection()

        for(i in output3!!.size / 2..output3!!.size / 2+400){
            Log.i(TAG, "output3:"+ i+ " " +output3!![i].toString())
        }

        val list : MutableList<BBox> = arrayListOf()
        val list3 = processData(output3!!, 13*13, list)
        val list2 = processData(output2!!, 26*26, list)
        val list1 = processData(output1!!, 52*52, list)
        return list
    }

    override fun close() {
        inferenceInterface?.close()
    }

    /**
     * 对图片进行缩放
     * @param bitmap
     * @param size
     * @return
     * @throws IOException
     */
    @Throws(IOException::class)
    override fun getScaleBitmap(bitmap: Bitmap, size: Int): Bitmap {
        origin_width = bitmap.width
        origin_height = bitmap.height

        val scaleWidth = size.toFloat() / origin_width
        val scaleHeight = size.toFloat() / origin_height

        scale = Math.min(scaleHeight,scaleWidth)

        val w = (scale * origin_width).toInt()
        val h = (scale * origin_height).toInt()
        Log.i(TAG, "" + w +" "+h)

        val matrix = Matrix()
        matrix.postScale(scale, scale)
        var pic = Bitmap.createBitmap(bitmap, 0, 0, origin_width, origin_height, matrix, true)
        Log.i(TAG, "" + pic.width +" "+pic.height)

        var newBitmap =Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        val paddingW = (size - w) / 2
        val paddingH = (size - h) / 2

        Log.i(TAG, pic.width.toString())
        Log.i(TAG, pic.height.toString())

        for(i in 0..pic.width-1){
            for(j in 0..pic.height-1){
                newBitmap.set(i+paddingW,j+paddingH,pic.get(i,j))
            }
        }
        return newBitmap
    }


    fun processData(array: FloatArray, boxNum:Int, list: MutableList<BBox>) : MutableList<BBox>{
        var beginIndex = 0
        for(i in 0..boxNum-1){

            Log.i(TAG,"beginIndex:"+beginIndex)

            val x = array[beginIndex]
            val y = array[beginIndex+1]
            val w = array[beginIndex+2]
            val h = array[beginIndex+3]

            //(x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var xmin = x - w /2
            var ymin = y - h /2
            var xmax = x + w /2
            var ymax = y + h /2

            Log.i(TAG, "1. xmin:" + xmin+" xmax:"+ xmax+" ymin:" + ymin+" ymax:"+ ymax)

            //(xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            val paddingW = (inputSize - (scale * origin_width).toInt()) / 2
            val paddingH = (inputSize - (scale * origin_width).toInt()) / 2

            xmin = (xmin - paddingW) / scale
            ymin = (ymin - paddingH) / scale
            xmax = (xmax - paddingW) / scale
            ymax = (ymax - paddingH) / scale

            Log.i(TAG, "2 .xmin:" + xmin+" xmax:"+ xmax+" ymin:" + ymin+" ymax:"+ ymax)
            if(xmin < 0 || xmax > origin_width || ymin < 0 || ymax > origin_height){
                beginIndex = (beginIndex + 85)
                continue
            }

            val max =findBiggest(array, beginIndex+5,beginIndex+5+79)
            val score = array[beginIndex+4] * max[0]
            Log.i(TAG,"3. score = "+score)

            if(score < 0.3){
                beginIndex = (beginIndex + 85)
                continue
            }

            val box =BBox(xmin, xmax, ymin, ymax,max[1].toInt(),score)
            list.add(box)

            Log.i(TAG, "4."+box.classId.toString())
            beginIndex = (beginIndex + 85)
        }

        Log.i("size", list.size.toString())
        return list
    }

    fun findBiggest(array: FloatArray, beginIndex:Int, endIndex:Int):Array<Float>{
        var max = 0.0f
        var index = 0
        for(i in beginIndex..endIndex){
            if(array[i] > max) {
                max = array[i]
                index = i - beginIndex
            }
        }
        Log.i(TAG, "max:" + max.toString())
        return arrayOf(max,index.toFloat())
    }

    override fun toString(): String {
        return "YoloDetection(TAG='$TAG', context=$context, modelFileName='$modelFileName', input_name='$input_name', output_names=${Arrays.toString(
            output_names
        )}, intValues=${Arrays.toString(intValues)}, inputSize=$inputSize, floatValues=${Arrays.toString(
            floatValues
        )}, logStats=$logStats, output1=${Arrays.toString(output1)}, output2=${Arrays.toString(
            output2
        )}, output3=${Arrays.toString(output3)}, origin_width=$origin_width, origin_height=$origin_height, scale=$scale, inferenceInterface=$inferenceInterface)"
    }
}