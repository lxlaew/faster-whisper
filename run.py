from faster_whisper import WhisperModel
import time
import os


def get_file_type(file_path):
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 音频格式
    audio_formats = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    # 视频格式
    video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    if file_ext in audio_formats:
        return 'audio'
    elif file_ext in video_formats:
        return 'video'
    else:
        return 'unknown'


def transcribe_media(media_path, 
                    model_size="medium", 
                    device="cuda", 
                    compute_type="float16", 
                    beam_size=5, 
                    language="zh", 
                    print_results=True):
    """
    转录音频或视频文件为文字
    
    Args:
        media_path (str): 音频或视频文件路径
        model_size (str): 模型大小 ("small", "medium", "large-v3")
        device (str): 运行设备 ("cuda", "cpu")
        compute_type (str): 计算类型 ("float16", "int8_float16", "int8")
        beam_size (int): 集束搜索大小
        language (str): 指定语言代码
        print_results (bool): 是否打印结果
    
    Returns:
        tuple: (segments, info, elapsed_time, file_type)
    """
    # 检查文件是否存在
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"媒体文件不存在: {media_path}")
    
    # 检测文件类型
    file_type = get_file_type(media_path)
    
    if print_results:
        if file_type == 'audio':
            print(f"🎵 检测到音频文件: {os.path.basename(media_path)}")
        elif file_type == 'video':
            print(f"🎬 检测到视频文件: {os.path.basename(media_path)}")
            print("ℹ️  将从视频中提取音频进行转录...")
        else:
            print(f"⚠️  未知文件格式: {os.path.basename(media_path)}")
            print("ℹ️  尝试作为媒体文件处理...")
    
    # 初始化模型
    if print_results:
        print(f"🔧 正在加载 {model_size} 模型...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # 记录开始时间
    start_time = time.time()
    
    # 转录媒体文件（faster-whisper 会自动处理视频文件的音频提取）
    segments, info = model.transcribe(media_path, beam_size=beam_size, language=language)
    
    if print_results:
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        print("开始转录...\n")
    
    # 实时处理和打印每个音频段
    segments_list = []
    segment_count = 0
    
    for segment in segments:
        segments_list.append(segment)
        segment_count += 1
        
        if print_results:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            # 可选：显示进度
            # print(f"  -> 已处理 {segment_count} 段")
    
    # 记录结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if print_results:
        # 计算分钟和秒数
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        file_type_emoji = "🎵" if file_type == 'audio' else "🎬" if file_type == 'video' else "📁"
        print(f"\n{file_type_emoji} 转录完成！共处理了 {segment_count} 个音频段")
        print(f"⏱️ 总耗时: {minutes}分钟{seconds}秒")
    
    return segments_list, info, elapsed_time, file_type


def main():
    """主函数"""
    # 支持音频和视频文件
    # media_file = "audio.mp3"
    # media_file = "D:/myproject/douyin_live_stream/xihuji/xihuji.mp3"

    # 视频文件示例
    media_file = "video.mp4"
    # media_file = "D:/myproject/douyin_live_stream/xihuji/xihuji.mp4"

    
    try:
        segments, info, elapsed_time, file_type = transcribe_media(
            media_path=media_file,
            model_size="medium",  # 可选: "small", "medium", "large-v3"
            device="cuda",        # 可选: "cuda", "cpu"
            compute_type="float16",  # 可选: "float16", "int8_float16", "int8"
            beam_size=5,
            language="zh",
            print_results=True
        )
        
        print(f"\n✅ 成功处理 {file_type} 文件")
        
    except Exception as e:
        print(f"❌ 转录过程中发生错误: {e}")


if __name__ == "__main__":
    main()