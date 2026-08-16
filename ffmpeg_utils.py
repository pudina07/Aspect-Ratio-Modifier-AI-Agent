"""
FFmpeg helper utilities for finding and running FFmpeg/FFprobe commands reliably on Windows.
"""
import os
import shutil
import subprocess
from pathlib import Path
import imageio_ffmpeg

def get_ffmpeg_exe() -> str:
    """
    Returns the absolute path to a working ffmpeg binary.
    Prioritizes local ./tools/ffmpeg.exe, then imageio_ffmpeg, then system PATH.
    """
    local_tool = Path(__file__).resolve().parent / "tools" / "ffmpeg.exe"
    if local_tool.exists():
        return str(local_tool)
    
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
        
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    raise FileNotFoundError("FFmpeg executable not found!")

def run_ffmpeg_command(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """
    Executes an ffmpeg command replacing 'ffmpeg' with the resolved binary.
    """
    exe = get_ffmpeg_exe()
    if args and args[0] == "ffmpeg":
        cmd = [exe] + args[1:]
    else:
        cmd = [exe] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)

if __name__ == "__main__":
    exe = get_ffmpeg_exe()
    print("Resolved FFmpeg binary:", exe)
    res = run_ffmpeg_command(["-version"])
    print("FFmpeg output line 1:", res.stdout.splitlines()[0] if res.stdout else "None")
