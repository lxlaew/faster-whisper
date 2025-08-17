#!/usr/bin/env python3
"""
语音转录客户端
支持实时接收服务端的转录结果
"""

import requests
import json
import time
import argparse
import sys
from typing import Optional
from urllib.parse import urlencode


class ASRClient:
    """语音转录客户端"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        
    def test_connection(self) -> bool:
        """测试服务器连接"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 连接成功: {data.get('message', '服务器正常')}")
                return True
            else:
                print(f"❌ 服务器响应异常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def transcribe_file(
        self,
        media_path: str,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "zh",
        save_to_file: Optional[str] = None
    ):
        """转录本地文件"""
        
        print(f"🎯 开始转录文件: {media_path}")
        print(f"📋 参数: model={model_size}, device={device}, language={language}")
        print("=" * 60)
        
        # 构建请求参数
        params = {
            "media_path": media_path,
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "language": language
        }
        
        url = f"{self.server_url}/asr?{urlencode(params)}"
        
        try:
            # 发送 SSE 请求
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # 处理流式响应
            segments = []
            total_text = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # 去掉 'data: ' 前缀
                        self._handle_event(data, segments)
                        
                        # 收集文本用于保存
                        if data.get('type') == 'segment':
                            total_text += data.get('text', '') + "\n"
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  解析JSON失败: {e}")
            
            # 保存结果到文件
            if save_to_file and total_text.strip():
                self._save_to_file(save_to_file, total_text.strip(), segments)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
        except KeyboardInterrupt:
            print("\n\n⏹️  用户中断转录")
        except Exception as e:
            print(f"❌ 未知错误: {e}")
    
    def upload_and_transcribe(
        self,
        file_path: str,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "zh",
        save_to_file: Optional[str] = None
    ):
        """上传文件并转录"""
        
        print(f"📤 上传并转录文件: {file_path}")
        print(f"📋 参数: model={model_size}, device={device}, language={language}")
        print("=" * 60)
        
        try:
            # 准备文件上传
            with open(file_path, 'rb') as f:
                files = {'file': (file_path, f, 'application/octet-stream')}
                params = {
                    "model_size": model_size,
                    "device": device,
                    "compute_type": compute_type,
                    "beam_size": beam_size,
                    "language": language
                }
                
                url = f"{self.server_url}/asr/upload"
                
                # 发送上传请求
                response = requests.post(url, files=files, params=params, stream=True, timeout=300)
                response.raise_for_status()
                
                # 处理流式响应
                segments = []
                total_text = ""
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            self._handle_event(data, segments)
                            
                            if data.get('type') == 'segment':
                                total_text += data.get('text', '') + "\n"
                                
                        except json.JSONDecodeError as e:
                            print(f"⚠️  解析JSON失败: {e}")
                
                # 保存结果
                if save_to_file and total_text.strip():
                    self._save_to_file(save_to_file, total_text.strip(), segments)
                    
        except FileNotFoundError:
            print(f"❌ 文件不存在: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 上传失败: {e}")
        except KeyboardInterrupt:
            print("\n\n⏹️  用户中断转录")
        except Exception as e:
            print(f"❌ 未知错误: {e}")
    
    def _handle_event(self, data: dict, segments: list):
        """处理服务端事件"""
        event_type = data.get('type')
        
        if event_type == 'start':
            file_type_emoji = "🎵" if data.get('file_type') == 'audio' else "🎬"
            print(f"{file_type_emoji} 开始处理: {data.get('file_name')}")
            print()
            
        elif event_type == 'language_detected':
            lang = data.get('language')
            prob = data.get('language_probability', 0)
            print(f"🗣️  检测语言: {lang} (置信度: {prob:.3f})")
            print()
            
        elif event_type == 'segment':
            start = data.get('start_time', 0)
            end = data.get('end_time', 0)
            text = data.get('text', '')
            segment_id = data.get('segment_id', 0)
            
            # 实时显示转录结果
            print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")
            
            # 保存到列表
            segments.append({
                'id': segment_id,
                'start': start,
                'end': end,
                'text': text
            })
            
        elif event_type == 'complete':
            total_segments = data.get('total_segments', 0)
            elapsed_time = data.get('elapsed_time', 0)
            file_type = data.get('file_type', 'unknown')
            
            print()
            print("=" * 60)
            file_type_emoji = "🎵" if file_type == 'audio' else "🎬" if file_type == 'video' else "📁"
            print(f"{file_type_emoji} 转录完成！")
            print(f"📊 总段数: {total_segments}")
            print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
            
        elif event_type == 'error':
            error_msg = data.get('error_message', '未知错误')
            print(f"❌ 服务端错误: {error_msg}")
    
    def _save_to_file(self, file_path: str, text: str, segments: list):
        """保存结果到文件"""
        try:
            # 保存纯文本
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"💾 文本已保存到: {file_path}")
            
            # 保存详细信息到JSON文件
            json_file = file_path.rsplit('.', 1)[0] + '_detailed.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_segments': len(segments),
                    'segments': segments,
                    'full_text': text
                }, f, ensure_ascii=False, indent=2)
            print(f"📄 详细信息已保存到: {json_file}")
            
        except Exception as e:
            print(f"⚠️  保存文件失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音转录客户端")
    parser.add_argument("media_path", help="媒体文件路径")
    parser.add_argument("--server", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--model", default="medium", choices=["small", "medium", "large-v3"], help="模型大小")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="设备类型")
    parser.add_argument("--compute-type", default="float16", choices=["float16", "int8_float16", "int8"], help="计算类型")
    parser.add_argument("--beam-size", type=int, default=5, help="集束搜索大小")
    parser.add_argument("--language", default="zh", help="语言代码")
    parser.add_argument("--upload", action="store_true", help="上传文件到服务器")
    parser.add_argument("--save", help="保存结果到文件")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = ASRClient(args.server)
    
    # 测试连接
    if not client.test_connection():
        print("💡 请确保服务器正在运行: python run.py")
        sys.exit(1)
    
    print()
    
    # 执行转录
    if args.upload:
        client.upload_and_transcribe(
            file_path=args.media_path,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
            save_to_file=args.save
        )
    else:
        client.transcribe_file(
            media_path=args.media_path,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
            save_to_file=args.save
        )


if __name__ == "__main__":
    main()
