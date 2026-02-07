import os
from PIL import Image
import numpy as np

# 导入我们创建的API
from infer_images import infer_images


def main():
    """
    示例：使用API同时推理六张图片
    """
    # 配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), "config", "test_simple_md4allDDa_nuscenes.yaml")
    
    # 输出路径
    output_path = os.path.join(os.path.dirname(__file__), "output_api_example")
    
    # 示例1：从文件加载六张图片进行推理
    print("示例1: 从文件加载六张图片")
    
    # 假设我们有六张图片在resources目录
    # 这里使用同一张图片作为示例，实际使用时替换为不同的图片路径
    image_folder = os.path.join(os.path.dirname(__file__), "resources")
    image_files = [
        os.path.join(image_folder, "n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799608112460.jpg")
        for _ in range(6)
    ]
    
    # 加载图片
    images = [Image.open(img_path) for img_path in image_files]
    filenames = [f"example_image_{i}" for i in range(6)]
    
    # 执行推理
    depth_maps = infer_images(
        images=images,
        config_path=config_path,
        output_path=output_path,
        filenames=filenames,
        device='cuda'  # 或 'cpu'
    )
    
    print(f"成功处理 {len(depth_maps)} 张图像")
    print(f"深度图形状: {depth_maps[0].shape}")
    
    # 示例2：直接使用numpy数组进行推理
    print("\n示例2: 使用numpy数组")
    
    # 创建六张随机彩色图像作为示例
    # 实际使用时，这里应该是从其他程序接口获取的图像数据
    np_images = []
    for i in range(6):
        # 创建随机RGB图像 (H, W, 3)，范围0-255
        np_img = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        np_images.append(np_img)
    
    # 执行推理，不保存结果
    depth_maps_np = infer_images(
        images=np_images,
        config_path=config_path,
        output_path=None,  # 不保存结果
        filenames=[f"np_image_{i}" for i in range(6)],
        device='cuda'
    )
    
    print(f"成功处理 {len(depth_maps_np)} 张numpy数组图像")
    print(f"深度图形状: {depth_maps_np[0].shape}")
    
    # 示例3：带时间信息的推理
    print("\n示例3: 带时间信息的推理")
    
    # 假设我们知道每张图像的时间信息
    daytimes = ["day", "night", "day", "night", "day", "night"]
    
    depth_maps_time = infer_images(
        images=images,
        config_path=config_path,
        output_path=output_path,
        daytimes=daytimes,
        filenames=[f"time_image_{i}" for i in range(6)],
        device='cuda'
    )
    
    print(f"成功处理 {len(depth_maps_time)} 张带时间信息的图像")


if __name__ == "__main__":
    main()
