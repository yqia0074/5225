import requests
import base64
import json
import uuid
import os

def encode_image_to_base64(image_path):
    """将图像文件编码为base64字符串"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_pose_detection_json(base_url, image_path):
    """测试姿态检测JSON API"""
    print(f"Testing JSON API with image: {image_path}")
    
    # 编码图像
    image_base64 = encode_image_to_base64(image_path)
    
    # 创建请求数据
    request_data = {
        "id": str(uuid.uuid4()),
        "image": image_base64
    }
    
    # 发送POST请求
    try:
        response = requests.post(
            f"{base_url}/api/pose_detection",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ JSON API Success!")
            print(f"  Request ID: {result['id']}")
            print(f"  Persons detected: {result['count']}")
            print(f"  Bounding boxes: {len(result['boxes'])}")
            print(f"  Keypoints groups: {len(result['keypoints'])}")
            print(f"  Processing times:")
            print(f"    Preprocess: {result['speed_preprocess']:.4f}s")
            print(f"    Inference: {result['speed_inference']:.4f}s")
            print(f"    Postprocess: {result['speed_postprocess']:.4f}s")
            return True
        else:
            print(f"✗ JSON API Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ JSON API Error: {str(e)}")
        return False

def test_pose_detection_image(base_url, image_path, output_path):
    """测试姿态检测图像标注API"""
    print(f"Testing Image API with image: {image_path}")
    
    # 编码图像
    image_base64 = encode_image_to_base64(image_path)
    
    # 创建请求数据
    request_data = {
        "id": str(uuid.uuid4()),
        "image": image_base64
    }
    
    # 发送POST请求
    try:
        response = requests.post(
            f"{base_url}/api/pose_detection_annotation",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Image API Success!")
            print(f"  Request ID: {result['id']}")
            
            # 保存标注后的图像
            annotated_image_data = base64.b64decode(result['image'])
            with open(output_path, 'wb') as f:
                f.write(annotated_image_data)
            print(f"  Annotated image saved to: {output_path}")
            return True
        else:
            print(f"✗ Image API Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Image API Error: {str(e)}")
        return False

def test_health_check(base_url):
    """测试健康检查端点"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Health Check: {result}")
            return True
        else:
            print(f"✗ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health Check Error: {str(e)}")
        return False

def main():
    # 配置
    base_url = "http://localhost:60000"
    test_image = "./model3-yolol/test.jpg"  # 使用模型文件夹中的测试图像
    output_image = "./test_annotated_output.jpg"
    
    print("=== CloudPose API Test ===")
    print(f"Base URL: {base_url}")
    print(f"Test Image: {test_image}")
    print()
    
    # 检查测试图像是否存在
    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        print("Please make sure the test image exists.")
        return
    
    # 测试健康检查
    print("1. Testing Health Check...")
    test_health_check(base_url)
    print()
    
    # 测试JSON API
    print("2. Testing Pose Detection JSON API...")
    test_pose_detection_json(base_url, test_image)
    print()
    
    # 测试图像标注API
    print("3. Testing Pose Detection Image API...")
    test_pose_detection_image(base_url, test_image, output_image)
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()