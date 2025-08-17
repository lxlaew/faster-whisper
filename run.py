from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
import time
import os
import json
import asyncio
from typing import Optional, AsyncGenerator
import tempfile
import uvicorn
from pathlib import Path


app = FastAPI(title="语音转录服务", description="基于 faster-whisper 的实时语音转录 API")

# 全局模型缓存
_model_cache = {}


def get_file_type(file_path):
    """获取文件类型"""
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


def get_model(model_size="medium", device="cuda", compute_type="float16"):
    """获取或创建模型实例（带缓存）"""
    cache_key = f"{model_size}_{device}_{compute_type}"
    
    if cache_key not in _model_cache:
        print(f"🔧 正在加载 {model_size} 模型...")
        _model_cache[cache_key] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        print(f"✅ 模型 {model_size} 加载完成")
    
    return _model_cache[cache_key]


async def transcribe_media_stream(
    media_path: str,
    model_size: str = "medium",
    device: str = "cuda", 
    compute_type: str = "float16",
    beam_size: int = 5,
    language: str = "zh"
) -> AsyncGenerator[str, None]:
    """流式转录媒体文件，实时返回结果"""
    
    # 检查文件是否存在
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"媒体文件不存在: {media_path}")
    
    # 检测文件类型
    file_type = get_file_type(media_path)
    
    # 发送开始信息
    start_info = {
        "type": "start",
        "file_name": os.path.basename(media_path),
        "file_type": file_type,
        "timestamp": time.time()
    }
    yield f"data: {json.dumps(start_info, ensure_ascii=False)}\n\n"
    
    try:
        # 获取模型
        model = get_model(model_size, device, compute_type)
        
        # 记录开始时间
        start_time = time.time()
        
        # 转录媒体文件
        segments, info = model.transcribe(media_path, beam_size=beam_size, language=language)
        
        # 发送语言检测信息
        lang_info = {
            "type": "language_detected",
            "language": info.language,
            "language_probability": float(info.language_probability),
            "timestamp": time.time()
        }
        yield f"data: {json.dumps(lang_info, ensure_ascii=False)}\n\n"
        
        # 实时处理每个音频段
        segment_count = 0
        
        for segment in segments:
            segment_count += 1
            
            # 发送转录段结果
            segment_data = {
                "type": "segment",
                "segment_id": segment_count,
                "start_time": float(segment.start),
                "end_time": float(segment.end),
                "text": segment.text,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(segment_data, ensure_ascii=False)}\n\n"
            
            # 让出控制权，允许其他任务执行
            await asyncio.sleep(0.01)
        
        # 计算总耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 发送完成信息
        complete_info = {
            "type": "complete",
            "total_segments": segment_count,
            "elapsed_time": elapsed_time,
            "file_type": file_type,
            "timestamp": time.time()
        }
        yield f"data: {json.dumps(complete_info, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        # 发送错误信息
        error_info = {
            "type": "error",
            "error_message": str(e),
            "timestamp": time.time()
        }
        yield f"data: {json.dumps(error_info, ensure_ascii=False)}\n\n"


@app.get("/")
async def root():
    """API 根目录"""
    return {"message": "🎵 语音转录服务运行中", "version": "1.0.0"}


@app.get("/asr")
async def transcribe_by_path(
    media_path: str = Query(..., description="媒体文件的完整路径"),
    model_size: str = Query("medium", description="模型大小 (small/medium/large-v3)"),
    device: str = Query("cuda", description="设备类型 (cuda/cpu)"),
    compute_type: str = Query("float16", description="计算类型 (float16/int8_float16/int8)"),
    beam_size: int = Query(5, description="集束搜索大小"),
    language: str = Query("zh", description="语言代码")
):
    """通过文件路径进行语音转录 (Server-Sent Events)"""
    
    return StreamingResponse(
        transcribe_media_stream(
            media_path=media_path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            language=language
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# 【预留方案】直接上传文件到服务端，进行转录
@app.post("/asr/upload")
async def transcribe_by_upload(
    file: UploadFile = File(...),
    model_size: str = Query("medium", description="模型大小"),
    device: str = Query("cuda", description="设备类型"),
    compute_type: str = Query("float16", description="计算类型"),
    beam_size: int = Query(5, description="集束搜索大小"),
    language: str = Query("zh", description="语言代码")
):
    """通过文件上传进行语音转录 (Server-Sent Events)"""
    
    # 保存上传的文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_path = tmp_file.name
    
    try:
        return StreamingResponse(
            transcribe_media_stream(
                media_path=temp_path,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                language=language
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass


if __name__ == "__main__":
    print("🚀 启动语音转录服务...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )