import asyncio, struct, pyaudio, grpc
import audio_control_pb2 as pb2
import audio_control_pb2_grpc as pb2_grpc

RATE = 16000
CHUNK = 3200  # 0.2s chunks (tune this)

async def genMic():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            yield pb2.AudioChunk(pcm16=data, sample_rate=RATE)
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()

async def playTTS(replies):
    pa = pyaudio.PyAudio()
    out = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)
    async for t in replies:
        out.write(t.pcm16)

async def main():
    async with grpc.aio.insecure_channel('CLOUD_HOST:7000') as ch:
        stub = pb2_grpc.AudioIOStub(ch)
        replies = stub.Converse(genMic())
        await playTTS(replies)

if __name__ == '__main__':
    asyncio.run(main())