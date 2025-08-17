from faster_whisper import WhisperModel
import time

# model_size = "small"
model_size = "medium"
# model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# 记录开始时间
start_time = time.time()

segments, info = model.transcribe("D:/myproject/douyin_live_stream/xihuji/xihuji.mp3", beam_size=5, language="zh")
# segments, info = model.transcribe("audio.mp3", beam_size=5, language="zh")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# 记录结束时间并计算总耗时
end_time = time.time()
elapsed_time = end_time - start_time

# 计算分钟和秒数
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"\n总耗时: {minutes}分钟{seconds}秒")