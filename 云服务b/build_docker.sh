#!/bin/bash

# Docker镜像构建脚本
# 使用方法: ./build_docker.sh

IMAGE_NAME="cloudpose-service"
TAG="latest"

echo "开始构建Docker镜像..."
echo "镜像名称: $IMAGE_NAME:$TAG"

# 构建Docker镜像
docker build -t $IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功!"
    echo "镜像信息:"
    docker images | grep $IMAGE_NAME
    
    echo ""
    echo "🚀 运行容器命令:"
    echo "docker run -d -p 60000:60000 --name cloudpose-container $IMAGE_NAME:$TAG"
    
    echo ""
    echo "🔍 查看容器日志:"
    echo "docker logs cloudpose-container"
    
    echo ""
    echo "🛑 停止容器:"
    echo "docker stop cloudpose-container"
    
    echo ""
    echo "🗑️ 删除容器:"
    echo "docker rm cloudpose-container"
else
    echo "❌ Docker镜像构建失败!"
    exit 1
fi