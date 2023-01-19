# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_cn = QA("chinese_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    chinese_para = "阿里巴巴集團（NYSE：BABA、港交所：9988）创立于1999年，是一家提供電子商務線上交易平台的公司，服務範圍包括B2B貿易、網上零售、購物搜索引擎、第三方支付和雲計算服務。集團的子公司包括阿里巴巴B2B、淘寶網、天猫、一淘网、阿里雲計算、支付寶、螞蟻金服等。旗下的淘宝网和天猫在2012年销售额达到1.1万亿人民币，2015年度商品交易总额已经超过3万亿元人民币，是全球最大零售商。根據阿里巴巴集团向美国证券交易委员会提交的IPO招股书显示，雅虎持有阿里巴巴集團22.6%股權、軟銀持阿里集團34.4%股份，另管理層、僱員及其他投資者持股比例合共約為43%，當中馬雲持阿里巴巴集團約8.9%，蔡崇信持股為3.6%。至2012年九月底止的財政年度，以美國會計準則計算，阿里營業額按年增長74%至318.39億元（港元‧下同），盈利急升80%至37.75億元。2015年全年阿里巴巴营收146.01億美元，净利為74.94億美元。"

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a' , encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()

    # This block calls the prediction function and return the response
    try:        
        out = model_cn.predict(chinese_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 10:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
