package com.example.yolo.Tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Trace
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.set
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.IOException
import java.util.*
import android.R.attr.shape




class YoloDetection : Detector{

    private val TAG = "YoloDetection"
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

    private var inferenceInterface: TensorFlowInferenceInterface? = null

    constructor()


    companion object {
        fun create(assetManager: AssetManager, fileName : String,
                   inputSize:Int, outputSizes: Array<Int>,
                   inputName:String, outputNames: Array<String>): YoloDetection{
            var detection = YoloDetection()
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

            //val numClasses = operation1.output<Float>(0).shape().size(0) as Int
            Log.i("YoloDetection", ""+operation1.toString())
            Log.i("YoloDetection", ""+operation2.toString())
            Log.i("YoloDetection", ""+operation3.toString())
            return detection
        }
    }

    override fun detect(b: Bitmap) {
        val bitmap = getScaleBitmap(b, inputSize)
        Log.i(TAG, ""+bitmap.width+" "+bitmap.height)

        Trace.beginSection("detectImage")
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0,
            bitmap.getWidth(), bitmap.getHeight())

        for (i in 0 until intValues.size) {
            val `val` = intValues[i]
            floatValues!![i * 3 + 0] = (((`val` shr 16 and 0xFF)) / (255.0)).toFloat()
            floatValues!![i * 3 + 1] = (((`val` shr 8 and 0xFF)) /255.0).toFloat()
            floatValues!![i * 3 + 2] = (((`val` and 0xFF)) / (255.0)).toFloat()
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
    private fun getScaleBitmap(bitmap: Bitmap, size: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val scaleWidth = size.toFloat() / width
        val scaleHeight = size.toFloat() / height

        val scale = Math.min(scaleHeight,scaleWidth)

        val w = (scale * width).toInt()
        val h = (scale * height).toInt()
        Log.i(TAG, "" + w +" "+h)


        val matrix = Matrix()
        matrix.postScale(scale, scale)
        var pic = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
        Log.i(TAG, "" + pic.width +" "+pic.height)

        var newBitmap =Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        val paddingW = (size - w) / 2
        val paddingH = (size - h) / 2
        Log.i(TAG, "" + paddingW +" "+paddingH)

        for(i in 0..pic.width-1){
            for(j in 0..pic.height-1){
                newBitmap.set(i+paddingW,j+paddingH,pic.get(i,j))
            }
        }
        return newBitmap
    }

    override fun toString(): String {
        return "YoloDetection( " +
                "modelFileName='$modelFileName', " +
                "input_name='$input_name', " +
                "output_names=${Arrays.toString(output_names)}, " +
                "intValues=${Arrays.toString(intValues)}, " +
                "inputSize=$inputSize, " +
                "floatValues=${Arrays.toString(floatValues)}, " +
                "logStats=$logStats)"
    }

}