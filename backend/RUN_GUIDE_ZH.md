# 你的后端和语音部分怎么运行

在仓库根目录打开终端，第一次运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

然后打开 http://127.0.0.1:8000/voice-demo 。点击“음성 안내 시작”启动语音，再点左右方向按钮听韩语提示。图片上传测试可使用自己本地的一张 JPEG/PNG；图片不会作为项目文件上传 GitHub。

默认是 **mock 模拟模式**，固定返回左侧，不会真的识别图片。它用于让前端和后端先完成对接，不能作为模型检测成功或准确率的证据。要测试其他情况，停止服务器后运行：

```sh
CAKGO_MOCK_POSITION=right python -m uvicorn backend.app:app --port 8000
```

可选 left、center、right、missing。API说明：http://127.0.0.1:8000/docs 。

## 你可以向小组说明

“我已经实现了接收图片和目标名称的后端接口，按团队约定返回 JSON，计算目标的左中右位置，并提供韩语语音模块。现在通过 mock 模式测试接口；最终模型确认后切换适配器。相同提示4秒内不会重复，播放过程中不叠加语音。”

## 发给前端队友的对接说明

- POST `/api/detect`，用 FormData 传 `image` 和 `target`，不要手动设置 Content-Type。
- 取得 JSON 后调用 `voice.guide(result)`。
- 把 `voice/guidance.js` 加入前端项目，通过用户点击按钮调用 `voice.enable()`。
- 相机失败调用 `voice.error('camera')`；请求失败调用 `voice.error('network')`。
- 启动后端时用 `CAKGO_CORS_ORIGINS` 设置前端实际网址。
- 手机摄像头页面需要 HTTPS；同一Wi-Fi访问后端时可以绑定 `--host 0.0.0.0`，但 HTTPS 前端不能直接请求 HTTP 后端，需要同源代理或 HTTPS 部署。

## 发给模型队友的对接说明

适配器实现 `detect(image, target)`；image 是 RGB Pillow 图片，返回 `{'detections': [{'label': 'keys', 'confidence': 0.9, 'bbox': [x1,y1,x2,y2]}]}`。返回的列表应仅包含目标匹配，坐标使用原图像素。后端负责位置与文本。

现有 YOLO adapter 已兼容队友的 BGR 输入。运行真实 YOLO 需要额外安装其依赖并设置 `CAKGO_DETECTOR=yolo`；目前未下载权重、未验证模型速度，也不会把模型不支持的钥匙/钱包伪装成检测成功。

## 检查

```sh
python -m pytest backend/test_backend.py -q
node voice/test_guidance.cjs
```

第二条需要 Node.js。真实手机 iOS Safari、Android Chrome 的韩语音色和播放权限仍需设备实测。缺少韩语音色时会使用浏览器默认发音并保留屏幕文字。
