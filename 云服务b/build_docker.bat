@echo off
REM Docker镜像构建脚本 (Windows版本)
REM 使用方法: build_docker.bat

set IMAGE_NAME=cloudpose-service
set TAG=latest

echo 开始构建Docker镜像...
echo 镜像名称: %IMAGE_NAME%:%TAG%
echo.

REM 构建Docker镜像
docker build -t %IMAGE_NAME%:%TAG% .

if %ERRORLEVEL% EQU 0 (
    echo ✅ Docker镜像构建成功!
    echo 镜像信息:
    docker images | findstr %IMAGE_NAME%
    echo.
    echo 🚀 运行容器命令:
    echo docker run -d -p 60000:60000 --name cloudpose-container %IMAGE_NAME%:%TAG%
    echo.
    echo 🔍 查看容器日志:
    echo docker logs cloudpose-container
    echo.
    echo 🛑 停止容器:
    echo docker stop cloudpose-container
    echo.
    echo 🗑️ 删除容器:
    echo docker rm cloudpose-container
) else (
    echo ❌ Docker镜像构建失败!
    pause
    exit /b 1
)

pause