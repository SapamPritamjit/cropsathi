import pygame # Import pygame library for handling audio playback
import os # Import os library for handling file paths
import random # Import random library for generating random choice.
import asyncio # Import asyncio for asynchronous operations
import edge_tts # Import edge_tts for handling text-to-speech operations
from dotenv import dotenv_values # Import dotenv for handling environment variables from a .env file

# Load environment variables from a .env file
env_vars = dotenv_values(".env")
AssistantVoice = env_vars.get("AssistantVoice") # Get the AssistantVoice environment variable

# Asynchronous function to convert text to an audio file
async def TextToAudioFile(text) -> None:
    file_path = r"Data\speech.mp3" # Define the file path for the audio file

    if os.path.exists(file_path): # Check if the audio file already exists
        os.remove(file_path) # If it exists, remove it to avoid overwriting errors.

    # Create the communicate object to generate speech
    communicate = edge_tts.Communicate(text, AssistantVoice, pitch = '+5Hz', rate = '+13%')
    await communicate.save(r'Data\speech.mp3') # Save the generated speech as an MP3 file.

# Function to manage Text-to-Speech (TTS) functionality
def TTS(Text, func=lambda r=None: True):
    while True:
        try:
            # Convert text to an audio file asynchronously.
            asyncio.run(TextToAudioFile(Text))  

            # Initialize pygame mixer for audio playback
            pygame.mixer.init()

            # Load the generated speech file into pygame mixer
            pygame.mixer.music.load(r"Data\speech.mp3")
            pygame.mixer.music.play() # Play the audio

            # Loop until the audio is done playing or the function stops
            while pygame.mixer.music.get_busy():
                if func() == False: # Check if the external function returns false
                    break
                pygame.time.Clock().tick(10) # Limit the loop to 10 tricks per second
            return True # Return True if the audio played succeddfully
        
        except Exception as e:
            print(f"Error in TTS: {e}")

        finally:
            try:
                # Call the provided function with False to signal the end of TTS
                func(False)
                pygame.mixer.music.stop() # Stop the audio playback
                pygame.mixer.quit()

            except Exception as e: # Handle any exceptions during cleanup
                print(f"Error in finally block: {e}")
# Function to manage Text-to-Speech with additional responses for long text
def TextToSpeech(Text, func=lambda r=None: True):
    Data = str(Text).split(".") # Split the text by periods into a list of sentences 

    # List of predefined responses for cases where the text is too long
    responses = [
        "The rest of the result has been printed to the chat screen, kindly check it out sir.",
        "The rest of the text is now on the chat screen, sir, please check it.",
        "You can see the rest of the text on the chat screen, sir.",
        "The remaining part of the text is now on the chat screen, sir.",
        "Sir, you'll find more text on the chat screen for you to see.",
        "The rest of the answer is now on the chat screen, sir.",
        "Sir, please look at the chat screen, the rest of the answer is there.",
        "You'll find the complete answer on the chat screen, sir.",
        "The next part of the text is on the chat screen, sir.",
        "Sir, please check the chat screen for more information.",
        "There's more text on the chat screen for you, sir.",
        "Sir, take a look at the chat screen for additional text.",
        "You'll find more to read on the chat screen, sir.",
        "Sir, check the chat screen for the rest of the text.",
        "The chat screen has the rest of the text, sir.",
        "There's more to see on the chat screen, sir, please look.",
        "Sir, the chat screen holds the continuation of the text.",
        "You'll find the complete answer on the chat screen, kindly check it out sir.",
        "Please review the chat screen for the rest of the text, sir.",
        "Sir, look at the chat screen for the complete answer."
    ]

    # If the texxt is very long (more than 4 sentense and 250 characters), add a response message
    if len(Data) > 4 and len(Text) >= 250:
        TTS(" ".join(Text.split()[:2]) + "." + random.choice(responses), func)

    # Otherwise, just play the whole text
    else:
        TTS(Text, func)

# Main execution loop
if __name__ == "__main__":
    while True:
        #Prompt user for input and pass it to TextToSpeech function
        TextToSpeech(input("Enter the text: "))
        #TTS(input("Enter the text: ")) # If u want ai to read all the word then use these (morethat 250 words sentense)
