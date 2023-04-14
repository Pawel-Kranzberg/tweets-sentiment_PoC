#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:58:08 2019

@author: pk

See request and response examples below!
"""
import logging, pandas as pd
from flask import Flask, request, escape
app = Flask(__name__)


token = 'S8svkz6R7rPHz0wSZ2vhlLQnTb6eJCct'


@app.route('/sentiment', methods = ['GET', 'POST'])
def sentiment():
    logging.basicConfig(format = '%(levelname)s: %(message)s', level = logging.INFO)
    try:
        json = request.get_json()

        if escape(json.get('token', '')) != token:
             logging.info('Ignoring request without valid token')

        texts = json.get('texts', False)
#        return str(texts)
#    except:
#        pass

        if texts:
            texts = pd.Series(texts)
            texts2 = texts.str.strip() \
                .str.replace('(\s*RT @\w+: )|(\s+http\S+$)', '', regex = True) \
                .str.replace('\s*@\w+\s*', ' user ', regex = True) \
                .str.replace('\s*http\S+\s*', ' website ', regex = True) \
                .str.replace('\s*&amp;\s*', ' and ', regex = True)


            from fast_bert.prediction import BertClassificationPredictor

            predictor_sentiment = BertClassificationPredictor(model_path = './models/tweets_BERT_sentiment_04_rebalance_reweigh-2019-05-26_17-28-03.bin',
                                                    pretrained_path = 'bert-base-uncased',
                                                    label_path = './data/tweets_BERT_sentiment_04_rebalance_reweigh/',
                                                    multi_gpu = False,
                                                    multi_label = False)

            #predictions_sentiment = predictor_sentiment.predict_batch(texts)
            predictions_sentiment = pd.DataFrame([x[0] for x in predictor_sentiment.predict_batch(list(texts2))])

            predictor_negative = BertClassificationPredictor(model_path = './models/tweets_BERT_negatives_04_rebalance_reweigh-2019-05-26_19-09-30.bin',
                                        pretrained_path = 'bert-base-uncased',
                                        label_path = './data/tweets_BERT_negatives_04_rebalance_reweigh/',
                                        multi_gpu = False,
                                        multi_label = False)

            #predictions_negative = predictor_negative.predict_batch(texts)
            predictions_negative = pd.DataFrame([x[0] for x in predictor_negative.predict_batch(list(texts2))])

            df0 = pd.concat([texts, predictions_sentiment, predictions_negative], axis = 1)
            df0.columns = ['text', 'airline_sentiment', 'airline_sentiment_confidence', 'negativereason', 'negativereason_confidence']
            df0.loc[df0['airline_sentiment'] != 'negative', ['negativereason', 'negativereason_confidence']] = None

            return df0.to_json(orient = 'records')

    except:
        pass

#app.run(host = '0.0.0.0')

if __name__ == '__main__':
    app.run()

'''
### Request example
curl -X POST \
  http://0.0.0.0:5000/sentiment \
  -H 'Accept: */*' \
  -H 'Cache-Control: no-cache' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json' \
  -H 'Host: 0.0.0.0:5000' \
  -H 'Postman-Token: 1adb1862-6ee9-4c48-ae02-dabb6e11a7ce,76c37a2b-8fa1-4183-89ad-c0370be1e186' \
  -H 'User-Agent: PostmanRuntime/7.13.0' \
  -H 'accept-encoding: gzip, deflate' \
  -H 'cache-control: no-cache' \
  -H 'content-length: 2350' \
  -d '{"token": "S8svkz6R7rPHz0wSZ2vhlLQnTb6eJCct", "texts": ["My flight was cancelled", "I lost my luggage", "I love your airline", "@VirginAmerica  I flew from NYC to SFO last week and couldn'\''t fully sit in my seat due to two large gentleman on either side of me. HELP!", "I ❤️ flying @VirginAmerica. ☺️👍","@VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.", "@VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???", "@VirginAmerica I love this graphic. http://t.co/UT5GrRwAaA", "@VirginAmerica I love the hipster innovation. You are a feel good brand.", "@VirginAmerica will you be making BOS&gt", "@VirginAmerica you guys messed up my seating.. I reserved seating with my friends and you guys gave my seat away ... 😡 I want free internet", "@VirginAmerica status match program.  I applied and it'\''s been three weeks.  Called and emailed with no response.", "@VirginAmerica What happened 2 ur vegan food options?! At least say on ur site so i know I won'\''t be able 2 eat anything for next 6 hrs #fail", "@VirginAmerica do you miss me? Don'\''t worry we'\''ll be together very soon.", "@VirginAmerica amazing to me that we can'\''t get any cold air from the vents. #VX358 #noair #worstflightever #roasted #SFOtoBOS", "@VirginAmerica LAX to EWR - Middle seat on a red eye. Such a noob maneuver. #sendambien #andchexmix", "@VirginAmerica hi! I just bked a cool birthday trip with you, but i can'\''t add my elevate no. cause i entered my middle name during Flight Booking Problems 😢", "@VirginAmerica Are the hours of operation for the Club at SFO that are posted online current?", "@VirginAmerica help, left expensive headphones on flight 89 IAD to LAX today. Seat 2A. No one answering L&amp;F number at LAX!", "@VirginAmerica awaiting my return phone call, just would prefer to use your online self-service option :(", "@VirginAmerica this is great news!  America could start flights to Hawaii by end of year http://t.co/r8p2Zy3fe4 via @Pacificbiznews", "Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience http://t.co/Y7O0uNxTQP", "@VirginAmerica Moodlighting is the only way to fly! Best experience EVER! Cool and calming. 💜✈ #MoodlitMonday"
]}'

### Response example
[{"text":"My flight was cancelled","airline_sentiment":"negative","airline_sentiment_confidence":0.9994869232,"negativereason":"Cancelled Flight","negativereason_confidence":0.9967820644},{"text":"I lost my luggage","airline_sentiment":"negative","airline_sentiment_confidence":0.9997172952,"negativereason":"Lost Luggage","negativereason_confidence":0.9980018735},{"text":"I love your airline","airline_sentiment":"positive","airline_sentiment_confidence":0.9987784028,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica  I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!","airline_sentiment":"negative","airline_sentiment_confidence":0.9992637038,"negativereason":"Bad Flight","negativereason_confidence":0.6118991971},{"text":"I \u2764\ufe0f flying @VirginAmerica. \u263a\ufe0f\ud83d\udc4d","airline_sentiment":"positive","airline_sentiment_confidence":0.9976551533,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.","airline_sentiment":"positive","airline_sentiment_confidence":0.953574717,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???","airline_sentiment":"negative","airline_sentiment_confidence":0.999514699,"negativereason":"Flight Booking Problems","negativereason_confidence":0.9002380371},{"text":"@VirginAmerica I love this graphic. http:\/\/t.co\/UT5GrRwAaA","airline_sentiment":"positive","airline_sentiment_confidence":0.9985892177,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica I love the hipster innovation. You are a feel good brand.","airline_sentiment":"positive","airline_sentiment_confidence":0.9985671043,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica will you be making BOS&gt","airline_sentiment":"neutral","airline_sentiment_confidence":0.9984244108,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica you guys messed up my seating.. I reserved seating with my friends and you guys gave my seat away ... \ud83d\ude21 I want free internet","airline_sentiment":"negative","airline_sentiment_confidence":0.9997093081,"negativereason":"Customer Service Issue","negativereason_confidence":0.9575215578},{"text":"@VirginAmerica status match program.  I applied and it's been three weeks.  Called and emailed with no response.","airline_sentiment":"negative","airline_sentiment_confidence":0.99969697,"negativereason":"Customer Service Issue","negativereason_confidence":0.9990735054},{"text":"@VirginAmerica What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail","airline_sentiment":"negative","airline_sentiment_confidence":0.9997262359,"negativereason":"Can't Tell","negativereason_confidence":0.9965734482},{"text":"@VirginAmerica do you miss me? Don't worry we'll be together very soon.","airline_sentiment":"neutral","airline_sentiment_confidence":0.9394865632,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica amazing to me that we can't get any cold air from the vents. #VX358 #noair #worstflightever #roasted #SFOtoBOS","airline_sentiment":"negative","airline_sentiment_confidence":0.9997153878,"negativereason":"Bad Flight","negativereason_confidence":0.997574985},{"text":"@VirginAmerica LAX to EWR - Middle seat on a red eye. Such a noob maneuver. #sendambien #andchexmix","airline_sentiment":"neutral","airline_sentiment_confidence":0.9977706671,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica hi! I just bked a cool birthday trip with you, but i can't add my elevate no. cause i entered my middle name during Flight Booking Problems \ud83d\ude22","airline_sentiment":"neutral","airline_sentiment_confidence":0.9809655547,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica Are the hours of operation for the Club at SFO that are posted online current?","airline_sentiment":"neutral","airline_sentiment_confidence":0.9955105782,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica help, left expensive headphones on flight 89 IAD to LAX today. Seat 2A. No one answering L&amp;F number at LAX!","airline_sentiment":"negative","airline_sentiment_confidence":0.9981992841,"negativereason":"Customer Service Issue","negativereason_confidence":0.9989284873},{"text":"@VirginAmerica awaiting my return phone call, just would prefer to use your online self-service option :(","airline_sentiment":"negative","airline_sentiment_confidence":0.9995018244,"negativereason":"Customer Service Issue","negativereason_confidence":0.9990540147},{"text":"@VirginAmerica this is great news!  America could start flights to Hawaii by end of year http:\/\/t.co\/r8p2Zy3fe4 via @Pacificbiznews","airline_sentiment":"positive","airline_sentiment_confidence":0.9951466918,"negativereason":null,"negativereason_confidence":null},{"text":"Nice RT @VirginAmerica: Vibe with the moodlight from takeoff to touchdown. #MoodlitMonday #ScienceBehindTheExperience http:\/\/t.co\/Y7O0uNxTQP","airline_sentiment":"neutral","airline_sentiment_confidence":0.9975793958,"negativereason":null,"negativereason_confidence":null},{"text":"@VirginAmerica Moodlighting is the only way to fly! Best experience EVER! Cool and calming. \ud83d\udc9c\u2708 #MoodlitMonday","airline_sentiment":"positive","airline_sentiment_confidence":0.9984616041,"negativereason":null,"negativereason_confidence":null}]

'''
