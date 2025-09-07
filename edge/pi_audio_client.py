import asyncio, os, yaml, pyaudio, grpc
import audio_control_pb2 as pb2
import audio_control_pb2_grpc as pb2_grpc

with open("edge/config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

RATE  = int(CFG["audio"]["sample_rate"])
CHUNK = int(CFG["audio"]["chunk_size"])
HOST  = CFG["cloud"]["host"]
GRPC_PORT = int(CFG["cloud"]["grpc_port"])

def mic_stream():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=RATE,
        input=True,  frames_per_buffer=CHUNK,
        input_device_index=CFG["audio"]["input_device_index"]
            if CFG["audio"]["input_device_index"] is not None else None
    )
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield pb2.AudioChunk(pcm16=data, sample_rate=RATE)
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()

async def play_tts(replies):
    pa = pyaudio.PyAudio()
    out = pa.open(
        format=pyaudio.paInt16, channels=1, rate=RATE, output=True,
        output_device_index=CFG["audio"]["output_device_index"]
            if CFG["audio"]["output_device_index"] is not None else None
    )
    async for t in replies:
        out.write(t.pcm16)

async def run_once():
    target = f"{HOST}:{GRPC_PORT}"
    async with grpc.aio.insecure_channel(target) as ch:
        stub = pb2_grpc.AudioIOStub(ch)
        replies = stub.Converse(mic_stream())
        await play_tts(replies)

async def main():
    while True:
        try:
            await run_once()
        except Exception as e:
            print("[audio] error:", e, "— retry in 3s"); await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(main())