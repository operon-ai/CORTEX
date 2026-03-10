import os
import time
import httpx
import logging
import azure.cognitiveservices.speech as speechsdk
from typing import Optional

logger = logging.getLogger("cortex.audio")

class AzureAudio:
    """Unified utility for Azure STT and TTS."""

    def __init__(self):
        self.stt_endpoint = os.getenv("AZURE_OPENAI_AUDIO_ENDPOINT")
        self.stt_api_key = os.getenv("AZURE_TTS_KEY")
        
        self.tts_key = os.getenv("AZURE_TTS_KEY")
        self.tts_region = os.getenv("AZURE_TTS_REGION")
        
        # TTS Config
        if self.tts_key and self.tts_region:
            try:
                self.speech_config = speechsdk.SpeechConfig(subscription=self.tts_key, region=self.tts_region)
                self.speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural" # Premium voice
                self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
                self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=self.audio_config)
                print(f"[{time.strftime('%H:%M:%S')}] [AUDIO] Azure TTS initialized.", flush=True)
            except Exception as e:
                self.synthesizer = None
                print(f"[{time.strftime('%H:%M:%S')}] [AUDIO] Failed to init Azure TTS: {e}", flush=True)
        else:
            self.synthesizer = None
            logger.warning("Azure TTS credentials not fully set. TTS will be disabled.")
            print(f"[{time.strftime('%H:%M:%S')}] [AUDIO] Azure TTS disabled (missing keys).", flush=True)

        if self.stt_endpoint and self.stt_api_key:
            print(f"[{time.strftime('%H:%M:%S')}] [AUDIO] Azure STT initialized.", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [AUDIO] Azure STT disabled (missing keys).", flush=True)

    async def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Azure OpenAI Transcription endpoint."""
        if not self.stt_endpoint or not self.stt_api_key:
            logger.error("Azure STT credentials not set.")
            return "Error: STT credentials missing."

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return f"Error: File {audio_path} not found."

        try:
            async with httpx.AsyncClient() as client:
                with open(audio_path, "rb") as audio_file:
                    mime = "audio/webm" if audio_path.endswith(".webm") else "audio/wav"
                    files = {"file": (os.path.basename(audio_path), audio_file, mime)}
                    data = {"model": "gpt-4o-mini-transcribe"}
                    headers = {"api-key": self.stt_api_key}
                    
                    response = await client.post(
                        self.stt_endpoint,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=30.0
                    )
                
                if response.status_code == 200:
                    return response.json().get("text", "")
                else:
                    logger.error(f"Azure STT failed: {response.status_code} - {response.text}")
                    return f"Error: STT failed with status {response.status_code}"
        except Exception as e:
            logger.error(f"Exception during Azure STT: {e}")
            return f"Error: {str(e)}"

    def speak(self, text: str):
        """Synthesize and play speech using Azure TTS."""
        if not self.synthesizer:
            logger.error("TTS Synthesizer not initialized.")
            return

        try:
            result = self.synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Speech synthesized for text: {text[:50]}...")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(f"TTS Canceled: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"TTS Error code: {cancellation_details.error_code}")
                    logger.error(f"TTS Error details: {cancellation_details.error_details}")
        except Exception as e:
            logger.error(f"Exception during Azure TTS: {e}")
