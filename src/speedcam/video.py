import subprocess
import numpy as np


class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()
