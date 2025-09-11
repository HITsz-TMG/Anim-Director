import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import argparse
from typing import Optional, Union, List

class RealESRGAN:
    """
    A wrapper class for Real-ESRGAN image and video enhancement.
    Provides a simple interface to call both image and video enhancement functionality.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
    def enhance_image(
        self,
        input_path: str,
        output_path: str,
        model_name: str = 'RealESRGAN_x4plus_anime_6B',
        outscale: float = 4.0,
        denoise_strength: float = 0.5,
        face_enhance: bool = False,
        tile: int = 0,
        gpu_id: Optional[int] = None
    ) -> None:
        """
        Enhance images using Real-ESRGAN.
        
        Args:
            input_path: Path to input image or folder containing images
            output_path: Path to output directory
            model_name: Name of the model to use
            outscale: The final upsampling scale of the image
            denoise_strength: Denoise strength (0 for weak, 1 for strong)
            face_enhance: Whether to use GFPGAN for face enhancement
            tile: Tile size, 0 for no tile
            gpu_id: GPU device to use (can be None for auto-selection)
        """
        from inference_realesrgan import main as inference_image
        
        # Create argument namespace to simulate command line arguments
        args = argparse.Namespace(
            input=input_path,
            output=output_path,
            model_name=model_name,
            model_path=None,
            denoise_strength=denoise_strength,
            outscale=outscale,
            suffix='out',
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            face_enhance=face_enhance,
            fp32=False,
            alpha_upsampler='realesrgan',
            ext='auto',
            gpu_id=gpu_id
        )
        
        # Backup original sys.argv
        import sys
        original_argv = sys.argv.copy()
        
        try:
            # Create dummy sys.argv for the inference function
            sys.argv = ['inference_realesrgan.py']
            for k, v in vars(args).items():
                if v is not None:
                    if isinstance(v, bool) and v:
                        sys.argv.append(f'--{k}')
                    elif not isinstance(v, bool):
                        sys.argv.append(f'--{k}')
                        sys.argv.append(str(v))
            
            # Run the inference
            inference_image()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
    
    def enhance_video(
        self,
        input_path: str,
        output_path: str,
        model_name: str = 'realesr-animevideov3',
        outscale: float = 4.0,
        denoise_strength: float = 0.5,
        face_enhance: bool = False,
        tile: int = 0,
        fps: Optional[float] = None,
        extract_frame_first: bool = False
    ) -> None:
        """
        Enhance videos using Real-ESRGAN.
        
        Args:
            input_path: Path to input video
            output_path: Path to output directory
            model_name: Name of the model to use
            outscale: The final upsampling scale of the video
            denoise_strength: Denoise strength (0 for weak, 1 for strong)
            face_enhance: Whether to use GFPGAN for face enhancement
            tile: Tile size, 0 for no tile
            fps: Output video FPS (None to use the same as input)
            extract_frame_first: Whether to extract frames before processing
        """
        from inference_realesrgan_video import main as inference_video
        
        # Create argument namespace to simulate command line arguments
        args = argparse.Namespace(
            input=input_path,
            output=output_path,
            model_name=model_name,
            denoise_strength=denoise_strength,
            outscale=outscale,
            suffix='outx4',
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            face_enhance=face_enhance,
            fp32=False,
            fps=fps,
            ffmpeg_bin='ffmpeg',
            extract_frame_first=extract_frame_first,
            num_process_per_gpu=1,
            alpha_upsampler='realesrgan',
            ext='auto'
        )
        
        # Backup original sys.argv
        import sys
        original_argv = sys.argv.copy()
        
        try:
            # Create dummy sys.argv for the inference function
            sys.argv = ['inference_realesrgan_video.py']
            for k, v in vars(args).items():
                if v is not None:
                    if isinstance(v, bool) and v:
                        sys.argv.append(f'--{k}')
                    elif not isinstance(v, bool):
                        sys.argv.append(f'--{k}')
                        sys.argv.append(str(v))
            
            # Run the inference
            inference_video()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv



# if __name__ == "__main__":
#     upscaler = RealESRGAN()

#     # Enhance an image
#     upscaler.enhance_image(
#         input_path="Pipeline/videos/scene_1/clip_2/input_frame.jpg",
#         output_path="outputs"
#     )

#     # Enhance a video
#     upscaler.enhance_video(
#         input_path="Pipeline/videos/scene_1/clip_1/scene_1_clip_1_1.mp4",
#         output_path="outputs"
#     )