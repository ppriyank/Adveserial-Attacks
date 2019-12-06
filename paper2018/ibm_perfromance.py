from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

speech_to_text_api_key = None 
speech_to_text_authenticator = IAMAuthenticator(speech_to_text_api_key)
speech_to_text = SpeechToTextV1(authenticator=speech_to_text_authenticator)
speech_to_text.set_service_url('https://stream.watsonplatform.net/speech-to-text/api')


import os
path1 = "yey_00.wav"
path2 = "yey_10.wav"

with open(path1, 'rb') as audio_file:
    response = speech_to_text.recognize(audio=audio_file, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
    transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
    print(transcript)


with open(path2, 'rb') as audio_file:
    response = speech_to_text.recognize(audio=audio_file, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
    transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
    print(transcript)