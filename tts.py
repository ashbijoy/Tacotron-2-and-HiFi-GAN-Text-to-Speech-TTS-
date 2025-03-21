import pyttsx3

def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    engine.setProperty('rate', 150)    # Speed of speech (words per minute)
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Optional: Change voice (depends on system voices available)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[14].id)  # voices[0] for male, voices[1] for female (if available)
    
    # Convert text to speech
    engine.say(text)
    engine.runAndWait()


    

# Example usage
if __name__ == "__main__":
    sample_text = "Hello Rekha Ma'am, My name is Daniel this is a test of the text to speech system. This group consists of Ashwin Bijoy, Reethika Shree and Nivya Raju. They belong to the Data Science Department. Thank You for listening to me"
    text_to_speech(sample_text)