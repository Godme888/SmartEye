package com.example.smarteye

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectionResult
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraExecutor: ExecutorService

    private var frameCount = 0
    private var skipFrames = 2

    private var objectDetector: ObjectDetector? = null
    private var poseLandmarker: PoseLandmarker? = null

    @Volatile
    private var latestPoseResult: PoseLandmarkerResult? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private val vehicleClasses = setOf("car", "truck", "bus", "motorcycle")
    private val personClass = "person"

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 100
        private const val CAMERA_PERMISSION = Manifest.permission.CAMERA

        private const val MODEL_DIR = "models"
        private const val OBJECT_DETECTOR_MODEL = "efficientdet_lite0.tflite"
        private const val POSE_LANDMARKER_MODEL = "pose_landmarker_lite.task"

        private const val OBJECT_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
        private const val POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            initModels()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(CAMERA_PERMISSION),
                REQUEST_CAMERA_PERMISSION
            )
        }
    }

    private fun allPermissionsGranted(): Boolean {
        return ContextCompat.checkSelfPermission(
            baseContext,
            CAMERA_PERMISSION
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (allPermissionsGranted()) {
                initModels()
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.camera_permission_denied),
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun initModels() {
        CoroutineScope(Dispatchers.IO).launch {
            val modelDir = File(filesDir, MODEL_DIR)
            if (!modelDir.exists()) {
                modelDir.mkdirs()
            }

            val objectDetectorFile = File(modelDir, OBJECT_DETECTOR_MODEL)
            val poseLandmarkerFile = File(modelDir, POSE_LANDMARKER_MODEL)

            if (!objectDetectorFile.exists()) {
                downloadFile(OBJECT_DETECTOR_URL, objectDetectorFile)
            }
            if (!poseLandmarkerFile.exists()) {
                downloadFile(POSE_LANDMARKER_URL, poseLandmarkerFile)
            }

            try {
                val objectDetectorOptions = ObjectDetector.ObjectDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.CPU).build())
                    .setScoreThreshold(0.5f)
                    .setMaxResults(5)
                    .build()
                objectDetector = ObjectDetector.createFromFileAndOptions(
                    this@MainActivity,
                    objectDetectorFile.absolutePath,
                    objectDetectorOptions
                )

                val poseLandmarkerOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setDelegate(Delegate.CPU).build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setResultListener { result, _ ->
                        latestPoseResult = result
                    }
                    .setErrorListener { error ->
                        error.printStackTrace()
                    }
                    .build()
                poseLandmarker = PoseLandmarker.createFromFileAndOptions(
                    this@MainActivity,
                    poseLandmarkerFile.absolutePath,
                    poseLandmarkerOptions
                )

                mainHandler.post {
                    Toast.makeText(this@MainActivity, "Models loaded", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                mainHandler.post {
                    Toast.makeText(this@MainActivity, "Failed to load models", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun downloadFile(url: String, destFile: File) {
        try {
            val urlConnection = URL(url).openConnection()
            urlConnection.connectTimeout = 30000
            urlConnection.readTimeout = 60000

            val inputStream: InputStream = BufferedInputStream(urlConnection.getInputStream())
            val outputStream = FileOutputStream(destFile)

            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }

            outputStream.close()
            inputStream.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        frameCount++
                        if (frameCount % skipFrames == 0) {
                            processFrame(imageProxy)
                        } else {
                            imageProxy.close()
                        }
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    fun processFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        if (bitmap == null || objectDetector == null) {
            return
        }

        val mpImage = com.google.mediapipe.tasks.vision.core.MpImage.createFromBitmap(bitmap)
        val detectionResult = objectDetector?.detect(mpImage)

        val vehicles = mutableListOf<ObjectDetector.Detection>()
        var firstPerson: ObjectDetector.Detection? = null

        detectionResult?.detections()?.forEach { detection ->
            val category = detection.categories().firstOrNull()?.categoryName()?.lowercase()
            if (category != null) {
                if (vehicleClasses.contains(category)) {
                    vehicles.add(detection)
                } else if (category == personClass && firstPerson == null) {
                    firstPerson = detection
                }
            }
        }

        mainHandler.post {
            overlayView.updateVehicles(vehicles)
        }

        if (firstPerson != null && poseLandmarker != null) {
            val boundingBox = firstPerson!!.boundingBox()
            val personBitmap = cropBitmap(bitmap, boundingBox)
            if (personBitmap != null) {
                val personMpImage = com.google.mediapipe.tasks.vision.core.MpImage.createFromBitmap(personBitmap)
                poseLandmarker?.detectAsync(personMpImage, System.currentTimeMillis())
            }
        }

        latestPoseResult?.let { result ->
            mainHandler.post {
                overlayView.updatePoses(result, firstPerson?.boundingBox())
            }
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null
        val planes = image.planes

        if (image.format != ImageFormat.YUV_420_888) {
            return null
        }

        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val imageBytes = out.toByteArray()

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val matrix = Matrix()
        matrix.postRotate(90f)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun cropBitmap(bitmap: Bitmap, boundingBox: com.google.mediapipe.tasks.components.containers.BoundingBox): Bitmap? {
        val left = (boundingBox.left() * bitmap.width).toInt()
        val top = (boundingBox.top() * bitmap.height).toInt()
        val right = (boundingBox.right() * bitmap.width).toInt()
        val bottom = (boundingBox.bottom() * bitmap.height).toInt()

        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <= 0 || left < 0 || top < 0 || right > bitmap.width || bottom > bitmap.height) {
            return null
        }

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    fun getScaleRatio(): Float {
        val previewWidth = previewView.width.toFloat()
        val previewHeight = previewView.height.toFloat()
        val imageWidth = 480f
        val imageHeight = 640f
        return min(previewWidth / imageWidth, previewHeight / imageHeight)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        objectDetector?.close()
        poseLandmarker?.close()
    }
}