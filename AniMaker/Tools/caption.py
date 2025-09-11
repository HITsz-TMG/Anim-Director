import os
import sys
sys.path.append('Tools')
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

class VideoCaption:
    """
    A class for adding captions to videos.
    """
    
    def __init__(self):
        print("VideoCaption Initialized!")
    
    def add_caption(self, video_path, caption_text, output_path, font='Arial', fontsize=66, color='white', 
                   bg_color='transparent', position='bottom', opacity=0.7):
        """
        Add caption to the video.
        
        Args:
            output_path (str, optional): Path to save the output video. If None, adds '_captioned' to original filename.
            font (str, optional): Font of the caption text. Defaults to 'Arial'.
            fontsize (int, optional): Font size of the caption text. Defaults to 24.
            color (str, optional): Color of the caption text. Defaults to 'white'.
            bg_color (str, optional): Background color of the caption. Defaults to 'black'.
            position (str, optional): Position of the caption. Can be 'top', 'center', or 'bottom'. Defaults to 'bottom'.
            opacity (float, optional): Opacity of the caption background. Between 0 and 1. Defaults to 0.7.
            
        Returns:
            str: Path to the output video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not caption_text:
            raise ValueError("Caption text cannot be empty")
        
        # Default output path
        if output_path is None:
            filename, ext = os.path.splitext(video_path)
            output_path = f"{filename}_captioned{ext}"
        
        # Load the video
        video = VideoFileClip(video_path)
        
        # Create text clip for the caption
        txt_clip = TextClip(caption_text, fontsize=fontsize, color=color, font=font, bg_color=bg_color)
        
        # Set position of the caption
        if position == 'top':
            txt_clip = txt_clip.set_position(('center', 20))
        elif position == 'center':
            txt_clip = txt_clip.set_position('center')
        else:  # bottom
            txt_clip = txt_clip.set_position(('center', video.size[1] - txt_clip.size[1] - 20))
        
        # Set the duration of the text clip to match the video
        txt_clip = txt_clip.set_duration(video.duration)
        
        # Set opacity
        if opacity < 1:
            txt_clip = txt_clip.set_opacity(opacity)
        
        # Composite the video and text
        final_clip = CompositeVideoClip([video, txt_clip])
        
        # Write the output video
        final_clip.write_videofile(output_path)
        
        # Close the clips to free resources
        video.close()
        final_clip.close()
        
        return output_path


# if __name__ == "__main__":
#     # Video path and caption text
#     video_path = "Pipeline/videos/chosen/path1/scene_1_clip_1.mp4"
#     output_path = "Pipeline/videos/chosen/final/captioned_path1_scene_1_clip_1.mp4"
#     caption_text = "The Little Prince kneels by a seedling, hope in his eyes. You'll be beautiful."
    
#     # Create VideoCaption instance
#     captioner = VideoCaption()
    
#     # Example with custom settings
#     output = captioner.add_caption(
#         video_path = video_path,
#         caption_text = caption_text,
#         output_path = output_path
#     )
#     print(f"Custom captioned video saved to: {output}")
