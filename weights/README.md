# YOLOv8 权重目录

请将你训练好的 YOLOv8 权重文件（`.pt`）放在此目录下。

- **默认约定**：将权重文件命名为 `best.pt` 并放在本目录，即路径为 `backend/weights/best.pt`。
- **自定义路径**：在项目根目录的 `.env` 中设置 `YOLO_WEIGHTS_PATH`，例如：
  - `YOLO_WEIGHTS_PATH=weights/best.pt`（相对于 backend 目录）
  - 或使用绝对路径：`YOLO_WEIGHTS_PATH=D:/models/my_yolov8.pt`

放置好后，调用接口 `POST /image/detect` 并上传图片即可进行目标检测。
