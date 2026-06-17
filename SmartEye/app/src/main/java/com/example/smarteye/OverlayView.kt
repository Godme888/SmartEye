package com.example.smarteye

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.BoundingBox
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.min

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val rectPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val posePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val pointPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private var vehicles: List<ObjectDetector.Detection> = emptyList()
    private var poseResult: PoseLandmarkerResult? = null
    private var personBoundingBox: BoundingBox? = null

    private val POSE_CONNECTIONS = listOf(
        Pair(11, 12), Pair(11, 13), Pair(13, 15),
        Pair(12, 14), Pair(14, 16),
        Pair(23, 24), Pair(23, 25), Pair(25, 27),
        Pair(24, 26), Pair(26, 28),
        Pair(11, 23), Pair(12, 24),
        Pair(27, 29), Pair(29, 31), Pair(28, 30), Pair(30, 32)
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawVehicles(canvas)
        drawPose(canvas)
    }

    fun updateVehicles(detections: List<ObjectDetector.Detection>) {
        this.vehicles = detections
        invalidate()
    }

    fun updatePoses(result: PoseLandmarkerResult?, boundingBox: BoundingBox?) {
        this.poseResult = result
        this.personBoundingBox = boundingBox
        invalidate()
    }

    private fun drawVehicles(canvas: Canvas) {
        val scaleRatio = getScaleRatio()

        vehicles.forEach { detection ->
            val bbox = detection.boundingBox()
            val left = bbox.left() * scaleRatio
            val top = bbox.top() * scaleRatio
            val right = bbox.right() * scaleRatio
            val bottom = bbox.bottom() * scaleRatio

            canvas.drawRect(left, top, right, bottom, rectPaint)
        }
    }

    private fun drawPose(canvas: Canvas) {
        val poseLandmarks = poseResult?.poseLandmarks() ?: return
        if (poseLandmarks.isEmpty()) return

        val landmarks = poseLandmarks[0]
        val scaleRatio = getScaleRatio()
        val personBbox = personBoundingBox ?: return

        val personLeft = personBbox.left() * scaleRatio
        val personTop = personBbox.top() * scaleRatio
        val personWidth = (personBbox.right() - personBbox.left()) * scaleRatio
        val personHeight = (personBbox.bottom() - personBbox.top()) * scaleRatio

        POSE_CONNECTIONS.forEach { (startIdx, endIdx) ->
            if (startIdx < landmarks.size && endIdx < landmarks.size) {
                val startLandmark = landmarks[startIdx]
                val endLandmark = landmarks[endIdx]

                val startVisibility = startLandmark.visibility() ?: 0f
                val endVisibility = endLandmark.visibility() ?: 0f

                if (startVisibility > 0.5f && endVisibility > 0.5f) {
                    val startX = personLeft + startLandmark.x() * personWidth
                    val startY = personTop + startLandmark.y() * personHeight
                    val endX = personLeft + endLandmark.x() * personWidth
                    val endY = personTop + endLandmark.y() * personHeight

                    canvas.drawLine(startX, startY, endX, endY, posePaint)
                }
            }
        }

        landmarks.forEachIndexed { index, landmark ->
            val visibility = landmark.visibility() ?: 0f
            if (visibility > 0.5f) {
                val x = personLeft + landmark.x() * personWidth
                val y = personTop + landmark.y() * personHeight
                canvas.drawCircle(x, y, 8f, pointPaint)
            }
        }
    }

    private fun getScaleRatio(): Float {
        val previewWidth = width.toFloat()
        val previewHeight = height.toFloat()
        val imageWidth = 480f
        val imageHeight = 640f
        return min(previewWidth / imageWidth, previewHeight / imageHeight)
    }
}