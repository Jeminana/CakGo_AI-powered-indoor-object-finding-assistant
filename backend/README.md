# Backend Server — Member 3

## Component Purpose
The `backend/` component provides the REST API server (FastAPI/Python) that receives camera frames from the frontend, coordinates with the `detection/` engine, calculates the relative spatial position of the target item, composes guidance messages, and returns a structured response to the mobile client.

## Assigned Team Member
- **Member 3**
  - **Initial Responsibility:** Backend server setup, API endpoint design, mock detection responses, and voice guidance coordination.
  - **Post-Testing Responsibility:** Full backend, API, model adapter, and voice pipeline integration.

## Independent Work via Mock Responses
Member 3 can establish the server endpoints and provide simulated mock detection responses immediately, allowing Member 2 (Frontend) and Member 3 (Voice) to progress without waiting for final model selection.

## Tasks & Development Priorities
1. **API Server Skeleton:** Build FastAPI server with CORS middleware, request logging, and health check (`GET /api/health`).
2. **Mock Detection Endpoint:** Implement `POST /api/detect` returning realistic mock JSON payloads following [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md).
3. **Spatial Calculation Logic:** Compute target position based on bounding box horizontal center (`cx = (x1 + x2) / 2`) relative to image width:
   - `cx < 1/3 * width` → `"left"`
   - `1/3 * width <= cx <= 2/3 * width` → `"center"`
   - `cx > 2/3 * width` → `"right"`
4. **Message Formatting:** Compose Korean directional feedback:
   - Right: `"찾는 물건이 화면 오른쪽에 보입니다."`
   - Center: `"찾는 물건이 화면 가운데에 보입니다."`
   - Left: `"찾는 물건이 화면 왼쪽에 보입니다."`
   - Missing: `"현재 카메라 화면에서 찾는 물건이 보이지 않습니다."`
5. **Detector Integration:** Once Member 1 & 4 finalize the model choice, replace mock logic with the unified `detection/` adapter.

## API Specification
Conforms strictly to [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md):
- **`GET /api/health`** → `{"status": "ok"}`
- **`POST /api/detect`**
  - **Request:** `multipart/form-data` with `image` (JPEG/PNG) and `target` (string).
  - **Response (200 OK):**
    ```json
    {
      "target": "keys",
      "detected": true,
      "position": "left",
      "image_width": 640,
      "image_height": 480,
      "detections": [
        {"label": "keys", "confidence": 0.89, "bbox": [50, 180, 180, 290]}
      ],
      "message": "찾는 물건이 화면 왼쪽에 보입니다."
    }
    ```
  - **Response (Target Not Found):**
    ```json
    {
      "target": "keys",
      "detected": false,
      "position": null,
      "image_width": 640,
      "image_height": 480,
      "detections": [],
      "message": "현재 카메라 화면에서 찾는 물건이 보이지 않습니다."
    }
    ```

## Instructions for AI Coding Agents
1. Work within `backend/`.
2. Strictly maintain backward compatibility with [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md).
3. Ensure uploaded images are processed ephemerally in memory without storing private photos to disk.
4. Keep the detector adapter abstracted so swapping between mock, YOLO, and Grounding DINO is clean and configurable.
