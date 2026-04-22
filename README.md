# deep-audio-vibe

Manual Docker run instructions (host port 8001 -> container port 8000).

## Build Image

```powershell
docker build -t deep-audio-vibe .
```

## Run Container On Host Port 8001

```powershell
docker rm -f deep-audio-vibe 2>$null
docker run -d --name deep-audio-vibe -p 8001:8000 deep-audio-vibe
```

## Verify

```powershell
docker ps --filter "name=deep-audio-vibe" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
Invoke-RestMethod -Uri "http://127.0.0.1:8001/"
```

## App URL

- Web UI: http://127.0.0.1:8001/app
- Health: http://127.0.0.1:8001/

## Dockerfile Port Configuration Check

- EXPOSE is set to `8000`.
- Runtime server binds to `0.0.0.0:8000` in CMD.

This is correct. Keep container/internal port at 8000 and change only the host-side mapping (`-p 8001:8000`).
