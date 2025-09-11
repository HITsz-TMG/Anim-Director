import cv2
import os

def extract_video_segment_and_last_frame(video_path, duration_seconds, output_video_path=None, output_frame_path=None):
    """
    截取视频的前x秒，并保存截取片段的最后一帧
    
    Args:
        video_path (str): 输入视频路径
        duration_seconds (float): 要截取的时长（秒）
        output_video_path (str): 输出视频路径，如果为None则不保存视频
        output_frame_path (str): 输出最后一帧图片路径，如果为None则使用默认路径
    
    Returns:
        bool: 操作是否成功
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算要截取的帧数
    target_frames = int(fps * duration_seconds)
    target_frames = min(target_frames, total_frames)  # 不超过视频总帧数
    
    print(f"视频信息: FPS={fps}, 总帧数={total_frames}, 分辨率={width}x{height}")
    print(f"截取前 {duration_seconds} 秒，共 {target_frames} 帧")
    
    # 初始化视频写入器（如果需要保存视频）
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    last_frame = None
    frame_count = 0
    
    # 读取并处理帧
    while frame_count < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存视频帧（如果需要）
        if out:
            out.write(frame)
        
        # 更新最后一帧
        last_frame = frame.copy()
        frame_count += 1
    
    # 保存最后一帧
    if last_frame is not None:
        if output_frame_path is None:
            # 生成默认的输出路径
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_frame_path = f"{base_name}_last_frame_{duration_seconds}s.jpg"
        
        cv2.imwrite(output_frame_path, last_frame)
        print(f"最后一帧已保存到: {output_frame_path}")
    
    # 清理资源
    cap.release()
    if out:
        out.release()
        print(f"截取的视频已保存到: {output_video_path}")
    
    print(f"成功处理 {frame_count} 帧")
    return True

def main():
    """
    主函数示例
    """
    # 示例用法
    video_path = "Tools/Wan2.1/demo/outputs/85-3-3_outx4_captioned.mp4"  # 输入视频路径
    duration = 3.0  
    output_video = "Tools/Wan2.1/demo/outputs/85-3-3_outx4_captioned.mp4.mp4"  # 输出视频路径
    #output_frame = None
    output_frame = "Tools/Wan2.1/demo/outputs/85-3-3_outx4_captioned.mp4.jpg"  # 输出最后一帧路径
    
    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在")
        return
    
    # 执行截取操作
    success = extract_video_segment_and_last_frame(
        video_path=video_path,
        duration_seconds=duration,
        output_video_path=output_video,
        output_frame_path=output_frame
    )
    
    if success:
        print("视频处理完成!")
    else:
        print("视频处理失败!")

if __name__ == "__main__":
    main()
