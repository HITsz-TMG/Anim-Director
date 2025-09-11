import cv2
import os

def concatenate_videos(video1_path, video2_path, output_path):
    """
    拼接两个视频文件
    
    Args:
        video1_path (str): 第一个视频文件路径
        video2_path (str): 第二个视频文件路径
        output_path (str): 输出视频文件路径
    
    Returns:
        bool: 操作是否成功
    """
    # 打开第一个视频
    cap1 = cv2.VideoCapture(video1_path)
    if not cap1.isOpened():
        print(f"错误: 无法打开视频文件 {video1_path}")
        return False
    
    # 打开第二个视频
    cap2 = cv2.VideoCapture(video2_path)
    if not cap2.isOpened():
        print(f"错误: 无法打开视频文件 {video2_path}")
        cap1.release()
        return False
    
    # 获取第一个视频的属性作为输出视频的标准
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取第二个视频的属性
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频1信息: {width1}x{height1}, FPS={fps1:.2f}, 帧数={frames1}")
    print(f"视频2信息: {width2}x{height2}, FPS={fps2:.2f}, 帧数={frames2}")
    
    # 使用第一个视频的属性创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (width1, height1))
    
    if not out.isOpened():
        print(f"错误: 无法创建输出视频文件 {output_path}")
        cap1.release()
        cap2.release()
        return False
    
    total_frames_written = 0
    
    # 写入第一个视频的所有帧
    print("正在处理第一个视频...")
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)
        total_frames_written += 1
        
        if total_frames_written % 100 == 0:
            print(f"已处理第一个视频 {total_frames_written} 帧")
    
    print(f"第一个视频处理完成，共 {total_frames_written} 帧")
    
    # 写入第二个视频的所有帧
    print("正在处理第二个视频...")
    frames_from_video2 = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        
        # 如果第二个视频的分辨率与第一个不同，进行调整
        if width2 != width1 or height2 != height1:
            frame = cv2.resize(frame, (width1, height1))
        
        out.write(frame)
        total_frames_written += 1
        frames_from_video2 += 1
        
        if frames_from_video2 % 100 == 0:
            print(f"已处理第二个视频 {frames_from_video2} 帧")
    
    print(f"第二个视频处理完成，共 {frames_from_video2} 帧")
    
    # 清理资源
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"视频拼接完成! 输出文件: {output_path}")
    print(f"总共写入 {total_frames_written} 帧")
    
    return True

def get_video_info(video_path):
    """
    获取视频信息
    
    Args:
        video_path (str): 视频文件路径
    
    Returns:
        dict: 视频信息字典
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

def main():
    """
    主函数示例
    """
    # 示例用法
    video1_path = "Tools/Wan2.1/demo/scene_1_clip_4_2s_h1.mp4"  # 第一个视频路径
    video2_path = "Tools/Wan2.1/demo/scene_1_clip_4_3s_h2.mp4"  # 第二个视频路径
    output_path = "Tools/Wan2.1/demo/outputs/69-1-4.mp4"  # 输出视频路径
    
    # 检查输入文件是否存在
    if not os.path.exists(video1_path):
        print(f"错误: 视频文件 {video1_path} 不存在")
        return
    
    if not os.path.exists(video2_path):
        print(f"错误: 视频文件 {video2_path} 不存在")
        return
    
    # 显示视频信息
    print("=== 视频信息 ===")
    info1 = get_video_info(video1_path)
    info2 = get_video_info(video2_path)
    
    if info1:
        print(f"视频1: {info1['width']}x{info1['height']}, {info1['fps']:.2f}fps, "
              f"{info1['duration']:.2f}秒, {info1['frame_count']}帧")
    
    if info2:
        print(f"视频2: {info2['width']}x{info2['height']}, {info2['fps']:.2f}fps, "
              f"{info2['duration']:.2f}秒, {info2['frame_count']}帧")
    
    # 执行拼接操作
    print("\n=== 开始拼接视频 ===")
    success = concatenate_videos(video1_path, video2_path, output_path)
    
    if success:
        print("视频拼接成功!")
        # 显示输出视频信息
        output_info = get_video_info(output_path)
        if output_info:
            print(f"输出视频: {output_info['width']}x{output_info['height']}, "
                  f"{output_info['fps']:.2f}fps, {output_info['duration']:.2f}秒, "
                  f"{output_info['frame_count']}帧")
    else:
        print("视频拼接失败!")

if __name__ == "__main__":
    main()
