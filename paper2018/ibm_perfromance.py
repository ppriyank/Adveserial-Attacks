from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

speech_to_text_api_key = None
speech_to_text_authenticator = IAMAuthenticator(speech_to_text_api_key)
speech_to_text = SpeechToTextV1(authenticator=speech_to_text_authenticator)
speech_to_text.set_service_url('https://stream.watsonplatform.net/speech-to-text/api')



import os
# path1 = "yey_00.wav"
# path2 = "yey_10.wav"
# path3 = "yey_20.wav"

# audio_file1 = open(path1, 'rb')
# audio_file2 = open(path2, 'rb')
# audio_file3 = open(path3, 'rb')

# response = speech_to_text.recognize(audio=audio_file1, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
# transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
# print(transcript)

# response = speech_to_text.recognize(audio=audio_file2, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
# transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
# print(transcript)


# response = speech_to_text.recognize(audio=audio_file3, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
# transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
# print(transcript)

import os
import glob
import shutil

for file in glob.glob("attacked4/*.wav"):
	try:
		print(file)	
		audio_file = open(file, 'rb')
		response = speech_to_text.recognize(audio=audio_file, content_type='audio/wav', model='en-US_BroadbandModel').get_result()
		transcript = ''.join([sentence['alternatives'][0]['transcript'] for sentence in response['results']])
		print(transcript)
	except Exception as e:
		# os.remove(file) 
		continue 


