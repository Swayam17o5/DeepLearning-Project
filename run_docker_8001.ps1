$ErrorActionPreference = "Stop"

$imageName = "deep-audio-vibe"
$containerName = "deep-audio-vibe"
$hostPort = 8001
$containerPort = 8000

Write-Host "Building Docker image: $imageName"
docker build --no-cache -t $imageName .

Write-Host "Removing old container (if exists): $containerName"
docker rm -f $containerName 2>$null | Out-Null

Write-Host "Starting container on host port $hostPort -> container port $containerPort"
docker run -d --name $containerName -p "${hostPort}:${containerPort}" $imageName

Write-Host "Container status:"
docker ps --filter "name=$containerName" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"

Write-Host "Health check URL: http://127.0.0.1:$hostPort/"
Write-Host "Web app URL:     http://127.0.0.1:$hostPort/app"
