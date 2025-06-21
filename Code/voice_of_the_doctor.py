# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

# Import required libraries
import os
import subprocess
import platform
from gtts import gTTS

# Function to Convert Text to Speech Using gTTS
def text_to_speech_with_gtts(input_text, output_filepath="doctor_voice.mp3"):
    language = "en"
    
    # Convert text to speech
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath)
    
    print(f"Audio saved to {output_filepath}")

    # Auto-play the generated audio file based on OS (only if running outside Gradio)
    if __name__ == "__main__":  
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', output_filepath])
            elif os_name == "Windows":  # Windows
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath}").PlaySync();'])
            elif os_name == "Linux":  # Linux
                subprocess.run(['aplay', output_filepath])  # Alternative: use 'mpg123' or 'ffplay'
            else:
                raise OSError("Unsupported operating system")
        except Exception as e:
            print(f"An error occurred while trying to play the audio: {e}")

    # Return the audio file path for Gradio playback
    return output_filepath

# Test the function (only when running standalone)
if __name__ == "__main__":
    input_text = "Hi, this is AI Doctor speaking with you!"
    text_to_speech_with_gtts(input_text, "doctor_voice.mp3")
