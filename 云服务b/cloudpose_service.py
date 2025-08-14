from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import time
import threading
from typing import List, Dict, Any
import logging
from PIL import Image
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="CloudPose API", description="Pose Detection Web Service", version="1.0.0")

# 全局模型变量
model = None

# Request data model
class ImageRequest(BaseModel):
    id: str
    image: str  # base64 encoded image

# Response data model
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    probability: float

class PoseResponse(BaseModel):
    id: str
    count: int
    boxes: List[BoundingBox]
    keypoints: List[List[List[float]]]
    speed_preprocess: float
    speed_inference: float
    speed_postprocess: float

class ImageResponse(BaseModel):
    id: str
    image: str  # base64 encoded annotated image

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model
    try:
        log.info("Loading YOLO pose detection model...")
        model = YOLO('./model3-yolol/yolo11l-pose.pt')
        log.info("Model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise e

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode to OpenCV image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    try:
        # Encode image as JPEG format
        _, buffer = cv2.imencode('.jpg', image)
        # Convert to base64 string
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

def process_pose_detection(image: np.ndarray) -> tuple:
    """Process pose detection and return results"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocessing timing
    start_preprocess = time.time()
    # Preprocessing steps can be added here, currently using original image directly
    preprocess_time = time.time() - start_preprocess
    
    # Inference timing
    start_inference = time.time()
    results = model(image)
    inference_time = time.time() - start_inference
    
    # Postprocessing timing
    start_postprocess = time.time()
    
    boxes = []
    keypoints_list = []
    count = 0
    
    for result in results:
        if result.boxes is not None:
            # Process bounding boxes
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                boxes.append(BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    probability=float(conf)
                ))
        
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            # Process keypoints
            for i in range(len(result.keypoints.xy)):
                person_keypoints = []
                keypoints_xy = result.keypoints.xy[i].cpu().numpy()
                keypoints_conf = result.keypoints.conf[i].cpu().numpy()
                
                for j, (x, y) in enumerate(keypoints_xy):
                    conf = keypoints_conf[j] if j < len(keypoints_conf) else 0.0
                    person_keypoints.append([float(x), float(y), float(conf)])
                
                keypoints_list.append(person_keypoints)
                count += 1
    
    postprocess_time = time.time() - start_postprocess
    
    return boxes, keypoints_list, count, preprocess_time, inference_time, postprocess_time

def annotate_image(image: np.ndarray) -> np.ndarray:
    """Annotate keypoints on image"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Copy image to avoid modifying original
    annotated_image = image.copy()
    
    results = model(image)
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            for i in range(len(result.keypoints.xy)):
                keypoints_xy = result.keypoints.xy[i].cpu().numpy()
                keypoints_conf = result.keypoints.conf[i].cpu().numpy()
                
                # Draw keypoints
                for k, (x, y) in enumerate(keypoints_xy):
                    if k < len(keypoints_conf) and keypoints_conf[k] > 0.5:
                        cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(annotated_image, str(k), (int(x) + 5, int(y) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw connections (COCO format human keypoint connections)
                connections = [
                    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],  # shoulders to arms
                    [5, 11], [6, 12], [11, 12],  # shoulders to hips
                    [11, 13], [12, 14], [13, 15], [14, 16],  # hips to legs
                    [0, 1], [0, 2], [1, 3], [2, 4]  # head keypoints
                ]
                
                for connection in connections:
                    if (connection[0] < len(keypoints_xy) and connection[1] < len(keypoints_xy) and
                        connection[0] < len(keypoints_conf) and connection[1] < len(keypoints_conf) and
                        keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5):
                        
                        p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                        p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))
                        cv2.line(annotated_image, p1, p2, (0, 0, 255), 2)
    
    return annotated_image

@app.post("/api/pose_detection", response_model=PoseResponse)
async def pose_detection_json(request: ImageRequest):
    """Pose detection JSON API endpoint"""
    try:
        log.info(f"Processing pose detection request with ID: {request.id}")
        
        # Decode base64 image
        image = base64_to_image(request.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Process pose detection
        boxes, keypoints_list, count, preprocess_time, inference_time, postprocess_time = process_pose_detection(image)
        
        # Build response
        response = PoseResponse(
            id=request.id,
            count=count,
            boxes=boxes,
            keypoints=keypoints_list,
            speed_preprocess=preprocess_time,
            speed_inference=inference_time,
            speed_postprocess=postprocess_time
        )
        
        log.info(f"Successfully processed request {request.id}, detected {count} persons")
        return response
        
    except Exception as e:
        log.error(f"Error processing request {request.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/pose_detection_annotation")
async def pose_detection_image(request: ImageRequest):
    """Pose detection image annotation HTML API endpoint"""
    from fastapi.responses import HTMLResponse
    try:
        log.info(f"Processing pose detection annotation request with ID: {request.id}")
        
        # Decode base64 image
        image = base64_to_image(request.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Annotate image
        annotated_image = annotate_image(image)
        
        # Encode to base64
        annotated_base64 = image_to_base64(annotated_image)
        
        # Build HTML response
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Annotation Results - {request.id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .image-container {{ text-align: center; margin: 30px 0; }}
                .annotated-image {{ max-width: 100%; height: auto; border: 2px solid #007bff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .download-section {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .download-link {{ display: inline-block; padding: 10px 20px; background-color: #28a745; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }}
                .download-link:hover {{ background-color: #218838; }}
                .back-link {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .back-link:hover {{ background-color: #0056b3; }}
                .base64-section {{ background-color: #f1f3f4; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .base64-text {{ word-break: break-all; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; background-color: white; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Image Annotation Results</h1>
                <div class="summary">
                    <h3>Processing Summary</h3>
                    <p><strong>Request ID:</strong> {request.id}</p>
                    <p><strong>Processing Status:</strong> ✅ Successfully Completed</p>
                    <p><strong>Processing Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="image-container">
                    <h3>Annotated Image</h3>
                    <img src="data:image/jpeg;base64,{annotated_base64}" alt="Annotated image" class="annotated-image">
                </div>
                
                <div class="download-section">
                    <h3>Download Options</h3>
                    <p>You can right-click on the image and select "Save As" to save the annotated image.</p>
                    <a href="data:image/jpeg;base64,{annotated_base64}" download="annotated_{request.id}.jpg" class="download-link">Download Annotated Image</a>
                </div>
                
                <div class="base64-section">
                    <h3>Base64 Encoded Data</h3>
                    <p>Below is the Base64 encoded data of the annotated image (can be used for API calls):</p>
                    <div class="base64-text">{annotated_base64[:200]}...(Truncated, full data length: {len(annotated_base64)} characters)</div>
                </div>
                
                <a href="/" class="back-link">Back to Home</a>
            </div>
        </body>
        </html>
        """
        
        log.info(f"Successfully processed annotation request {request.id}")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        log.error(f"Error processing annotation request {request.id}: {str(e)}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - {request.id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .error {{ background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; }}
                .back-link {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Processing Error</h1>
                <div class="error">
                    <h3>Error Information</h3>
                    <p><strong>Request ID:</strong> {request.id}</p>
                    <p><strong>Error Details:</strong> {str(e)}</p>
                </div>
                <a href="/" class="back-link">Back to Home</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from fastapi.responses import HTMLResponse
    status = "healthy" if model is not None else "unhealthy"
    model_status = "Loaded" if model is not None else "Not Loaded"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CloudPose API - Health Check</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .status {{ color: #28a745; font-weight: bold; font-size: 18px; }}
            .info {{ background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CloudPose API Health Check</h1>
            <div class="status">✅ Service Status: {status}</div>
            <div class="info">
                <strong>Model Status:</strong> {model_status}
            </div>
            <div class="info">
                <strong>Check Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/")
async def root():
    """Root endpoint"""
    from fastapi.responses import HTMLResponse
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CloudPose API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .status { color: #28a745; font-weight: bold; }
            .endpoint { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }
            .method { color: #007bff; font-weight: bold; }
            .form-section { margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CloudPose API</h1>
            <p class="status">✅ Service is running</p>
            <h2>Available API Endpoints:</h2>
            <div class="endpoint">
                <span class="method">GET</span> <a href="/health">/health</a> - Health check
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/pose_detection - Pose detection JSON API
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/pose_detection_annotation - Pose detection image annotation API
            </div>
            
            <div class="form-section">
                <h3>Test API</h3>
                <p><strong>Note:</strong> POST endpoints require JSON requests containing base64 encoded images.</p>
                <div class="form-group">
                    <label for="imageFile">Select image file:</label>
                    <input type="file" id="imageFile" accept="image/*">
                </div>
                <div class="form-group">
                    <label for="requestId">Request ID:</label>
                    <input type="text" id="requestId" value="test-" placeholder="Enter request ID">
                </div>
                <div class="form-group">
                    <button onclick="generateRequestJson()">Generate Request JSON</button>
                    <button onclick="testPoseDetection()">Test Pose Detection</button>
                    <button onclick="testPoseAnnotation()">Test Image Annotation</button>
                </div>
                <div id="result" style="margin-top: 20px;"></div>
            </div>
        </div>
        
        <script>
        function generateRequestJson() {
            const fileInput = document.getElementById('imageFile');
            const requestId = document.getElementById('requestId').value || 'test-' + Date.now();
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64 = e.target.result.split(',')[1];
                const requestJson = {
                    id: requestId,
                    image: base64
                };
                
                document.getElementById('result').innerHTML = 
                    '<h4>Request JSON Format:</h4>' +
                    '<pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">' + 
                    JSON.stringify(requestJson, null, 2) + 
                    '</pre>';
            };
            reader.readAsDataURL(fileInput.files[0]);
        }
        
        function testPoseDetection() {
            const fileInput = document.getElementById('imageFile');
            const requestId = document.getElementById('requestId').value || 'test-' + Date.now();
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64 = e.target.result.split(',')[1];
                
                fetch('/api/pose_detection', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        id: requestId,
                        image: base64
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerHTML = 
                        '<h4>Pose Detection Results (JSON Format):</h4>' +
                        '<pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">' + 
                        JSON.stringify(data, null, 2) + 
                        '</pre>';
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = '<h4>Error:</h4><pre>' + error + '</pre>';
                });
            };
            reader.readAsDataURL(fileInput.files[0]);
        }
        
        function testPoseAnnotation() {
            const fileInput = document.getElementById('imageFile');
            const requestId = document.getElementById('requestId').value || 'test-' + Date.now();
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const base64 = e.target.result.split(',')[1];
                
                fetch('/api/pose_detection_annotation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        id: requestId,
                        image: base64
                    })
                })
                .then(response => response.text())
                .then(data => {
                    document.getElementById('result').innerHTML = '<h4>Image Annotation Results:</h4><pre>' + data + '</pre>';
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = '<h4>Error:</h4><pre>' + error + '</pre>';
                });
            };
            reader.readAsDataURL(fileInput.files[0]);
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=60000)