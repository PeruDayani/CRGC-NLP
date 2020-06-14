#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import collections 
from collections import Counter

import api_helpers

# TO DO: Replace find_keywords(df) with api_helpers.findSignificantPhases everywhere

app = Flask(__name__)

def find_keywords(df):
	df = api_helpers.cleanData(df)
	df = api_helpers.findSignificantPhases(df)
	return df

@app.route('/')
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/model", methods=['GET', 'POST'])
def model():
	
	if request.method == 'POST':

		valid_input = True

		name = request.form['name']
		email = request.form['email']
		phone = request.form['phone']

		try:
			df = pd.read_csv(request.files.get('file'))
		except:
			return render_template('model.html', info="---", input_data="Invalid", model_report="Cannot generate")

		if (name=="") or (email=="") or (phone==""):
			path_info = api_helpers.extractPathInfo(df)

			if path_info == None:
				return render_template('model.html', info="Invalid", input_data="---", model_report="Cannot generate")

			name = path_info[0]
			email = path_info[1] 
			phone = path_info[2]


		info = "<br>" + name + "<br>" + email + "<br>" + phone + "<br>"
		

		keywords = find_keywords(df)
		model_results = api_helpers.runModel(df)

		final_results = ""
		record = 1

		for i in range(len(keywords)):

			keyword = keywords[i]
			col_pred_df, model_prediction = model_results[i]

			if model_prediction == 0:
				model_prediction = "No Muscle Present"
				print("Email Alert")
				api_helpers.emailAlert(email)
			else:
				model_prediction = "Muscle Present"

			record_html =  "<br> Record " + str(record) + "<br>"
			keywords_html = "<br> Keywords Found: <br>" + keyword.to_html(classes=["table-bordered", "table-striped", "table-hover"]) + "<br>"
			col_pred_html = "<br> Per Column Model Prediction :" + col_pred_df.to_html(classes=["table-bordered", "table-striped", "table-hover"]) + "<br>"
			model_html = "<br> Final Model Prediction : " + str(model_prediction) + "<br>"

			final_result =  record_html + keywords_html + col_pred_html + model_html + "<br><br>"
			final_results += final_result

			record += 1

		return render_template('model.html', info=info, input_data=df.to_html(classes=["table-bordered", "table-striped", "table-hover", "table-sm", "table-responsive"]), model_report=final_results)


	return render_template("model.html")


@app.route("/keywords_test", methods=['GET', 'POST'])
def keywords_test():

	if request.method == 'POST':

		name = "Peru Dayani"
		email = "xxx@gmail.com"
		phone = "xxx.xxx.xxxx"

		info = name + " <br> " + email + " <br> " + phone
		
		df = api_helpers.sample()
		results = find_keywords(df)
		result = results[0]

		return render_template('keywords_test.html', info=info, input=df.to_html(classes=["table-bordered", "table-striped", "table-hover", "table-sm", "table-responsive"]), output=result.to_html(classes=["table-bordered", "table-striped", "table-hover"]))

	return render_template("keywords_test.html")
	
@app.route("/keywords_input", methods=['GET', 'POST'])
def keywords_input():
	if request.method == 'POST':

		name = request.form['name']
		email = request.form['email']
		phone = request.form['phone']

		info = name + " <br> " + email + " <br> " + phone

		text1 = request.form['adText_Final_DX']
		text2 = request.form['adText_OP_Proc']
		text3 = request.form['adText_Path']
		text4 = request.form['adText_Phys_EX']
		text5 = request.form['adText_Remarks']
		text6 = request.form['adText_Scopes']
		text7 = request.form['adText_Surg_1']

		data = [text1, text2, text3, text4, text5, text6, text7]

		df = api_helpers.toDataframe(data)
		results = find_keywords(df)
		result = results[0]

		return render_template('keywords_input.html', info=info, input=df.to_html(classes=["table-bordered", "table-striped", "table-hover", "table-sm", "table-responsive"]), output=result.to_html(classes=["table-bordered", "table-striped", "table-hover"]))

	return render_template("keywords_input.html")

@app.route("/keywords_csv", methods=['GET', 'POST'])
def keywords_csv():

	if request.method == 'POST':

		name = request.form['name']
		email = request.form['email']
		phone = request.form['phone']

		info = name + " <br> " + email + " <br> " + phone
		
		df = pd.read_csv(request.files.get('file'))
		results = find_keywords(df)

		result = ""
		record = 1
		for table in results:
			result +=  "<br> Record " + str(record) + "<br>" + table.to_html(classes=["table-bordered", "table-striped", "table-hover"])
			record += 1

		return render_template('keywords_csv.html', info=info, input=df.to_html(classes=["table-bordered", "table-striped", "table-hover", "table-sm", "table-responsive"]), output=result)
	
	return render_template('keywords_csv.html')


@app.route('/find_keywords_test', methods=['GET','POST'])
def testKeywords():
	df = api_helpers.sample()
	keywords = find_keywords(df)

	result = {
		"input" : df.to_html(),
		"output": keywords.to_html()
	}
	
	result = {str(key): value for key, value in result.items()}
	
	return jsonify(result=result)


@app.route('/find_keywords_input', methods=['GET','POST'])
def findKeywordsInput():
	
	name = request.form['name']
	email = request.form['email']
	phone = request.form['phone']

	print(name + " " + email + " " + phone)

	text1 = request.form['text1']
	text2 = request.form['text2']
	text3 = request.form['text3']
	text4 = request.form['text4']
	text5 = request.form['text5']
	text6 = request.form['text6']
	text7 = request.form['text7']

	data = [text1, text2, text3, text4, text5, text6, text7]

	df = api_helpers.toDataframe(data)

	print(df.to_string())
	
	keywords = find_keywords(df)

	result = {
		"input" : df.to_html(),
		"output": keywords.to_html()
	}
	
	result = {str(key): value for key, value in result.items()}
	
	return jsonify(result=result)

if __name__ == "__main__":
	nltk.download('punkt')
	app.run(debug=True)



