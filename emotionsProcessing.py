
import http.client, urllib.request, urllib.parse, urllib.error, base64, sys
import pandas as pd
import numpy as np
import json
import gzip


def feedImageURL(url_link):
	headers = {
	    # Request headers. Replace the placeholder key below with your subscription key.
	    'Content-Type': 'application/json',
	    'Ocp-Apim-Subscription-Key': '5fd2f9f7fbbc4f448fbda4e2b4ad6cf0',
	}
	params = urllib.parse.urlencode({
	})
	# Replace the example URL below with the URL of the image you want to analyze.
	body = str({ 'url': str(url_link)  })
	try:
	    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
	    conn.request("POST", "/emotion/v1.0/recognize?%s" % params, body, headers)
	    response = conn.getresponse()
	    data = response.read()
	    print(data)
	    conn.close()
	    return data
	except Exception as e:
	    print(e.args)


def findScores(image_json):
	face_timeline = pd.DataFrame(columns = {'disgust', 'surprise', 'anger', 'sadness', 'happiness', 'neutral', 'fear', 'contempt', 'height', 'width', 'left', 'top'})
	image_json = image_json.decode()
	image_json = json.loads(image_json)
	num_faces = len(image_json)
	for idx in range(num_faces):
		emotion_score_values = image_json[idx]['scores']
		face_dimensions = image_json[idx]['faceRectangle']
		for key in face_dimensions.keys():
			emotion_score_values[key] = face_dimensions[key]
		face_timeline = face_timeline.append(emotion_score_values, ignore_index = True)
	return face_timeline


def returnEmotionDimensions(face_timeline):
	num_faces = len(face_timeline)
	df_emotions = face_timeline[['disgust', 'surprise', 'anger', 'sadness', 'happiness', 'neutral', 'fear', 'contempt']]
	top_emotion = df_emotions.idxmax(axis = 1)
	emotion_dimension_list = []
	for num in range(num_faces):
		emotion_dimension_json = {}
		best_emotion = top_emotion[num]
		emotion_dimension_json['emotion'] = best_emotion
		emotion_dimension_json['top'] = face_timeline.loc[num]['top']
		emotion_dimension_json['width'] = face_timeline.loc[num]['width']
		emotion_dimension_json['height'] = face_timeline.loc[num]['height']
		emotion_dimension_json['left'] = face_timeline.loc[num]['left']
		emotion_dimension_list.append(emotion_dimension_json)
	return emotion_dimension_list









