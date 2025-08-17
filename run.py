from faster_whisper import WhisperModel
import time
import os


def transcribe_audio(audio_path, 
                    model_size="medium", 
                    device="cuda", 
                    compute_type="float16", 
                    beam_size=5, 
                    language="zh", 
                    print_results=True):
    """
    转录音频文件为文字
    
    Args:
        audio_path (str): 音频文件路径
        model_size (str): 模型大小 ("small", "medium", "large-v3")
        device (str): 运行设备 ("cuda", "cpu")
        compute_type (str): 计算类型 ("float16", "int8_float16", "int8")
        beam_size (int): 集束搜索大小
        language (str): 指定语言代码
        print_results (bool): 是否打印结果
    
    Returns:
        tuple: (segments, info, elapsed_time)
    """
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 初始化模型
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # 记录开始时间
    start_time = time.time()
    
    # 转录音频
    segments, info = model.transcribe(audio_path, beam_size=beam_size, language=language)
    
    if print_results:
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        print("开始转录，实时输出结果...\n")
    
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
        print(f"\n转录完成！共处理了 {segment_count} 个音频段")
        print(f"总耗时: {minutes}分钟{seconds}秒")
    
    return segments_list, info, elapsed_time


def main():
    """主函数"""
    audio_file = "D:/myproject/douyin_live_stream/xihuji/xihuji.mp3"
    # audio_file = "audio.mp3"
    
    try:
        segments, info, elapsed_time = transcribe_audio(
            audio_path=audio_file,
            model_size="medium",  # 可选: "small", "medium", "large-v3"
            device="cuda",        # 可选: "cuda", "cpu"
            compute_type="float16",  # 可选: "float16", "int8_float16", "int8"
            beam_size=5,
            language="zh",
            print_results=True
        )        
        
    except Exception as e:
        print(f"转录过程中发生错误: {e}")


if __name__ == "__main__":
    main()