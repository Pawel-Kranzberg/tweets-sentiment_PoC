# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:06:16 2019

@author: PK
"""
## General purpose imports:
import pandas as pd
# from sklearn.model_selection import train_test_split

## For exploratory data analysis only:
import matplotlib.pyplot as plt, seaborn as sns, folium
from wordcloud import WordCloud, STOPWORDS


seed = 42
#sample_n = 1000
sns.set(style = 'ticks') # 'darkgrid')
palette = {'negative': 'red', 'neutral': 'orange', 'positive': 'chartreuse'} # 'cyan' 'magenta'

## Data import
#df0 = pd.read_csv('c:/Users/PK/OneDrive/Documents/NLP/_facts_Kaggle_Twitter_airline_sentiment/data/Tweets.csv')
df0 = pd.read_csv('./data/Tweets.csv')


### **Exploratory data analysis**

print(df0.shape, df0.dtypes, sep = '\n')

#df0.describe
#for x in df0:
#    print(x, df0[x].head(), sep = '\n')
print(df0.head())

print(df0.airline_sentiment.value_counts())
ax = sns.countplot(y = 'airline_sentiment', data = df0, order = palette.keys(), palette = palette)

print(df0.negativereason.value_counts())
ax0 = sns.countplot(y = 'negativereason', data = df0, order = df0.negativereason.value_counts().index)

print(df0[['airline_sentiment', 'airline_sentiment_gold', 'negativereason', 'negativereason_gold', 'tweet_id']].fillna('-').groupby(['airline_sentiment', 'airline_sentiment_gold', 'negativereason', 'negativereason_gold']).agg('count'))

## “What's in a name”
#print(df0.name.value_counts())
chart1, ax1 = plt.subplots()
sns.lineplot(data = df0.name.value_counts().cumsum().reset_index(drop = True))
ax1.set(xlabel = 'Accounts #', ylabel = 'Tweets #')
#sns.barplot(x = 'index', y = 'name', data = df0.name.value_counts().head(200).reset_index())
chart2, ax2 = plt.subplots()
sns.distplot(df0.name.value_counts(), kde = False, rug = True, axlabel = 'Tweets # per account')
ax2.set(ylabel = 'Accounts #')
df6 = df0.name.value_counts().head(20).reset_index().merge(df0, left_on = 'index', right_on = 'name', suffixes = ('_x', '')).sort_values('name_x', ascending = False)
df6 = pd.crosstab(df6.name, df6.airline_sentiment, margins = True).sort_values('All', ascending = False)
print('Top accounts', df6, sep = '\n')#.to_string)
del(df6)

## Temporal analysis
df0['tweet_created'] = pd.to_datetime(df0['tweet_created'], utc = True, errors = 'coerce')
df0['UTC_day_of_week'] = df0['tweet_created'].dt.dayofweek
df0['UTC_date'] = df0['tweet_created'].dt.date

df3 = df0[['UTC_date', 'airline_sentiment', 'airline']]
df3['n'] = 1
df4 = df3.groupby(['UTC_date', 'airline_sentiment']).agg('count').reset_index()
chart3, ax3 = plt.subplots()
sns.lineplot(x = 'UTC_date', y = 'n', data = df4, hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)
ax3.set_title('Overall airline sentiment over time')
# =============================================================================
# for a in sorted(df3['airline'].unique()):
#     print(a)
#     sns.relplot(x = 'UTC_date', y = 'n', data = df3[df3['airline'] == a].groupby(['UTC_date', 'airline_sentiment']).agg('count').reset_index(), kind = 'line', hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)
# =============================================================================
sns.relplot(col = 'airline', x = 'UTC_date', y = 'n', data = df3.groupby(['airline', 'UTC_date', 'airline_sentiment']).agg('count').reset_index(), kind = 'line', hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)
del(df3)

## Location analysis
v1 = df0['tweet_coord'].notna()
df1 = pd.DataFrame()
df1[['lat', 'lon']] = pd.DataFrame(df0['tweet_coord'][v1].apply(lambda x: [pd.eval(x)[0], pd.eval(x)[1]]).values.tolist(), index = df0.index[v1])
df0 = df0.merge(df1, left_index = True, right_index = True, how = 'left', validate = '1:1') # df0.join(df1, how = 'left')

for x in ['tweet_coord', 'tweet_location', 'user_timezone']:
    df0[x+ '_f'] = 0
    df0[x + '_f'][df0[x].notna() & (df0[x] != '[0.0, 0.0]')] = 1
    print(df0[x + '_f'].value_counts())

#sns.set(style = 'darkgrid')
#sns.scatterplot(x = 'lon', y = 'lat', hue = 'airline_sentiment', data = df0[v1])

df2 = df0[['airline_sentiment', 'lat', 'lon', 'airline', 'text']][v1].round(decimals = 0)
df2['n'] = 1
df2['text'] = df2['airline'] + ' - ' + df2['text']
df2.drop(columns = 'airline', inplace = True)
df2 = df2.groupby(['airline_sentiment', 'lat', 'lon']).agg({'n': 'count', 'text': list}).reset_index()
df2['n'] = df2['n'].astype(float)

m = folium.Map(location = [20, 0], tiles = 'Mapbox Bright', zoom_start = 2.5) # 'Mapbox Control Room'
for i in df2.index:
    folium.CircleMarker(
        location = [df2.loc[i, 'lat'], df2.loc[i, 'lon']],
        popup = df2.loc[i, 'text'],
        radius = df2.loc[i, 'n'],
        color = palette[df2.loc[i, 'airline_sentiment']],
        fill = True,
        fill_color = palette[df2.loc[i, 'airline_sentiment']]
    ).add_to(m)
m

## Scatterplot matrix of numeric and quasi-numeric variables
#sns.pairplot(df0.sort_values('airline_sentiment'))
sns.pairplot(df0, hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)

for x in ['tweet_coord_f', 'tweet_location_f', 'user_timezone_f']:
    df0[x] = df0[x].astype('category')

## Analysis of numeric variables
numeric_variables = list(df0.select_dtypes(include = 'number'))
#numeric_variables = list(df0.select_dtypes(include = ['number', 'datetimetz']))
#sample_df, _ = train_test_split(df0, train_size = sample_n, random_state = seed, stratify = df0[['airline', 'airline_sentiment']])
for y in numeric_variables:
    #sns.catplot(x = 'airline_sentiment', y = y, kind = 'swarm', data = sample_df)
    #sns.catplot(x = 'airline_sentiment', y = y, hue = 'airline_sentiment', kind = 'bar', data = df0.sort_values('airline_sentiment'))
    sns.catplot(x = 'airline_sentiment', y = y, kind = 'boxen', data = df0, order = palette.keys(), palette = palette)

## Analysis of categorical variables
#for x in['UTC_day_of_week', 'UTC_date']:
#    df0[x] = df0[x].astype('category')
df0['UTC_day_of_week'] = df0['UTC_day_of_week'].astype('category')
categorical_variables = sorted(set(df0.select_dtypes(include = ['object', 'category'])) - set(['UTC_date', 'airline_sentiment', 'airline_sentiment_gold', 'negativereason', 'negativereason_gold', 'text', 'name', 'tweet_coord', 'tweet_location']), key = str.lower)
charts_d = {}
for y in categorical_variables:
    print(y)
    df5 = pd.crosstab(df0[y], df0['airline_sentiment'], normalize = 0).reset_index().sort_values('negative', ascending = False)
    df5[y] = df5[y].astype('category')
    charts_d[y] = {}
    charts_d[y]['chart0'], charts_d[y]['ax0'] = plt.subplots()
    sns.barplot(y = y, x = 'value', data = df5.melt(id_vars = y), order = df5[y], \
                  hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)
    charts_d[y]['ax0'].set_title(y)
    charts_d[y]['chart1'], charts_d[y]['ax1'] = plt.subplots()
    sns.countplot(y = y, data = df0, order = df5[y], hue = 'airline_sentiment', hue_order = palette.keys(), palette = palette)

## Word clouds
df0['text2'] = df0['text'].str.strip() \
    .str.replace('(\s*RT @\w+: )|(\s+http\S+$)', '', regex = True) \
    .str.replace('\s*@\w+\s*', ' user ', regex = True) \
    .str.replace('\s*http\S+\s*', ' website ', regex = True) \
    .str.replace('\s*&amp;\s*', ' and ', regex = True)

for x in df0['airline'].unique():
    v2 = df0['airline'] == x
    for y in x.split(' '):
        df0['text2'][v2] = df0['text2'][v2].str.replace('\s*{}\s*'.format(y), ' airline ', regex = True)

df0['text2'] = df0['text2'].str.strip().str.lower()

#print('By airline and sentiment:')
#for x in palette.keys():
#    for y in sorted(df0['airline'].unique()):
#        plt.figure(figsize = (20, 20))
#        wordcloud = WordCloud(
#                              background_color = 'black',
#                              stopwords = set(STOPWORDS).union({'flight', 'user', 'website'}),
#                              max_words = 1000,
#                              max_font_size = 120,
#                              random_state = seed
#                            ).generate(str(df0['text2'][(df0['airline_sentiment'] == x) & (df0['airline'] == y)]))
#        plt.imshow(wordcloud)
#        plt.title(' '.join([x, y]), fontsize = 20)
#        plt.axis('off')
#        plt.show()

for k in ['airline_sentiment', 'airline', 'negativereason']:
    print('By sentiment and {}:'.format(k))
    for x in [x for x in palette.keys() if x in df0[df0[k].notna()]['airline_sentiment'].unique()]:
        for y in sorted(df0[k][df0[k].notna()].unique()):
            if sum((df0['airline_sentiment'] == x) & (df0[k] == y)) > 0:
                plt.figure(figsize = (20, 20))
                wordcloud = WordCloud(
                                      background_color = 'black',
                                      stopwords = set(STOPWORDS).union({'flight', 'user', 'website', 'airline'}),
                                      max_words = 1000,
                                      max_font_size = 110,
                                      random_state = seed
                                    ).generate(str(df0['text2'][(df0['airline_sentiment'] == x) & (df0[k] == y)]))
                plt.imshow(wordcloud)
                plt.title(' '.join([x, y]), fontsize = 20)
                plt.axis('off')
                plt.show()

### **Data engineering**
train_full_file = './data/train_full.csv'
train_negatives_file = './data/train_negatives.csv'
df0.to_csv(train_full_file, index = False)
df0[df0['negativereason'].notna()].to_csv(train_negatives_file, index = False)

### **Model training setup**

## Basic imports
import logging, sys
from pathlib import Path

## Sampling imports
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

## DL training imports
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from fast_bert.data import BertDataBunch
from fast_bert.learner import BertLearner

## Metrics imports
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torch import Tensor
from fast_bert.metrics import accuracy#, roc_auc, fbeta
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, multilabel_confusion_matrix


## Metrics functions
## Bad idea: Convert the metrics based on roc_auc_score to work on actual scores (and not on predicted labels from argmax) in y_pred. <- Not necessary, results are exactly the same!
def multiclass_roc_auc_score_macro(y_pred:Tensor, y_true:Tensor, average = 'macro', sample_weight = None):
    '''Based on https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659 '''
    y_pred = np.argmax(y_pred, axis = 1) #.numpy()
    #y_true = y_true.detach().cpu().numpy()
    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average = average, sample_weight = sample_weight)

# def multiclass_roc_auc_score_macro(y_pred:Tensor, y_true:Tensor, average = 'macro', sample_weight = None):
    # '''Based on https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659 '''
    # y_pred_mask = np.argmax(y_pred, axis = 1) 
    # lb = LabelBinarizer()
    # y_true = lb.fit_transform(y_true)
    # y_pred = y_pred * lb.transform(y_pred_mask)
    # return roc_auc_score(y_true, y_pred, average = average, sample_weight = sample_weight)

def multiclass_roc_auc_score_micro(y_pred:Tensor, y_true:Tensor):
    return multiclass_roc_auc_score_macro(y_pred, y_true, average = 'micro')

def F1_macro(y_pred:Tensor, y_true:Tensor, average = 'macro', sample_weight = None):
    y_pred = np.argmax(y_pred, axis = 1) #.numpy()
    #y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average = average, sample_weight = sample_weight)

def F1_micro(y_pred:Tensor, y_true:Tensor):
    return F1_macro(y_pred, y_true, average = 'micro')


### Training function
def train_model(
        labels_list,
        job_label,
        weights_column,
        input_file,
        labels_column,
        rebalance_data = False,
        reweigh_data = False
        ):
    ## Metrics functions that need labels upfront
    def confusion_matrix_overall(y_pred:Tensor, y_true:Tensor, labels:list = labels_list, sample_weight = None):
        y_pred = np.argmax(y_pred, axis = 1) #.numpy()
        #y_true = y_true.detach().cpu().numpy()
        return confusion_matrix(y_true, y_pred, labels = [i for i in range(len(labels))], sample_weight = sample_weight)

    def confusion_matrix_by_class(y_pred:Tensor, y_true:Tensor, labels:list = labels_list, sample_weight = None, samplewise = False):
        y_pred = np.argmax(y_pred, axis = 1) #.numpy()
        #y_true = y_true.detach().cpu().numpy()
        return multilabel_confusion_matrix(y_true, y_pred, labels = [i for i in range(len(labels))], sample_weight = sample_weight, samplewise = samplewise)

    def roc_auc_score_by_class(y_pred:Tensor, y_true:Tensor, labels:list = labels_list, average = 'micro', sample_weight = None):
        y_pred = np.argmax(y_pred, axis = 1).numpy()
        y_true = y_true.detach().cpu().numpy()
        roc_auc_score_d = {}
        for i in range(len(labels)):
            lb = LabelBinarizer()
            y_true_i = y_true.copy()
            y_true_i[y_true != i] = len(labels) + 1
            y_true_i = lb.fit_transform(y_true_i)
            y_pred_i = y_pred.copy()
            y_pred_i[y_pred != i] = len(labels) + 1
            y_pred_i = lb.transform(y_pred_i)
            roc_auc_score_d[labels[i]] = roc_auc_score(y_true_i, y_pred_i, average = average, sample_weight = sample_weight)
        return roc_auc_score_d

    def F1_by_class(y_pred:Tensor, y_true:Tensor, labels:list = labels_list, sample_weight = None):
        y_pred = np.argmax(y_pred, axis = 1) #.numpy()
        #y_true = y_true.detach().cpu().numpy()
        F1_by_class_d = {}
        for i in range(len(labels)):
            F1_by_class_d[labels[i]] = f1_score(y_true, y_pred, average = 'micro', labels = [i]) # pos_label = i,
        return F1_by_class_d
        # return f1_score(y_true, y_pred, average = None)


    args = {
        'job_label': job_label,
        'preprocess_data': {
                'preprocess_data': True,
                'rebalance_data': rebalance_data,
                'reweigh_data': reweigh_data,
                'reweigh_factor': 2.0,
                'weights_column': weights_column,
                'input_file': input_file,
                'training_share': 0.7,
                'validation_share': 0.2,
                'seed': seed
                },
        'data_labels': {
                'text': 'text2',
                'labels_column': labels_column,
                'labels_list': labels_list
                },
        'folders': {
                'log': Path('./logs/'),
                'data': Path('./data/'),
                'output': Path('./models/')
                },
        'BERT_model': 'bert-base-uncased',
        'lower_case': True,
        'max_sequence_length': 64, # 512 in BERT paper, but max tweet length in the data set is ~140 characters
        'training_batch_size': 32, #16,
        #"eval_batch_size": 32, #16,
        'learning_rate': 3e-5, #5e-6, # 3e-5 in BERT paper
        'training_epochs': 1, #4
        'warmup_schedule': 'warmup_cosine_hard_restarts',
        'warmup_proportion': 0.1,
        'gradient_accumulation_steps': 1,
        #"local_rank": -1,
        'FP16': False, # 16-bit fine-tuning (FP16) requires NVIDIA Volta (launched in 2018) or newer GPU. GTX1070 is earlier Pascal (2016).
        'FP16_loss_scale': 128,
        'metrics': {
                    'functions': {
                            #'FastBert roc_auc': roc_auc,
                            'FastBert accuracy': accuracy,
                            #'FastBert fbeta': fbeta,
                            'confusion_matrix_overall': confusion_matrix_overall,
                            'confusion_matrix_by_class': confusion_matrix_by_class,
                            'multiclass_roc_auc_score_macro': multiclass_roc_auc_score_macro,
                            'multiclass_roc_auc_score_micro': multiclass_roc_auc_score_micro,
                            'roc_auc_score_by_class': roc_auc_score_by_class,
                            'F1_macro': F1_macro,
                            'F1_micro': F1_micro,
                            'F1_by_class': F1_by_class
                            }
                }
    }

    for x in args['folders'].values():
        x.mkdir(exist_ok = True)
    data_folder = args['folders']['data'] / '{}{}'.format(args['job_label'], '/')
    data_folder.mkdir(exist_ok = True)

    ## Logging setup:
    run_start_time = pd.Timestamp.today(tz = 'UTC').strftime('%Y-%m-%d_%H-%M-%S')
    log_file = str(args['folders']['log']/'log-{}-{}.txt'.format(run_start_time, args['job_label']))

    logging.basicConfig(#filename = log_file, filemode = 'w', # filemode = 'a'
                        format = '%(name)s %(levelname)s %(asctime)s: %(message)s',
                        datefmt = '%Y-%m-%d_%H-%M-%S',
                        level = logging.INFO,
                        handlers = [
                                logging.StreamHandler(sys.stdout),
                                logging.FileHandler(log_file)
                                ])

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')

    logger = logging.getLogger('learner') # Logger for the learner object below.

    logging.info('Start: {}'.format(pd.Timestamp.today(tz = 'UTC')))
    logging.info(args)

    ### Device setup:
    torch.cuda.empty_cache()

    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        # import apex # https://github.com/NVIDIA/apex
        multi_gpu = True
    else:
        multi_gpu = False
    logging.info('CUDA device count: {}, multi_gpu: {}'.format(torch.cuda.device_count(), multi_gpu))

    ### Data preprocessing:
    if args['preprocess_data']['preprocess_data']:
        df0 = pd.read_csv(args['preprocess_data']['input_file'])
        columns_to_process = [args['data_labels']['text'], args['data_labels']['labels_column']]
        if args['preprocess_data']['reweigh_data']:
            columns_to_process.append(args['preprocess_data']['weights_column'])
        df0 = df0[columns_to_process].fillna(0)

        train_df, df0 = train_test_split(df0,
                                         train_size = float(args['preprocess_data']['training_share']),
                                         stratify = df0[args['data_labels']['labels_column']],
                                         random_state = args['preprocess_data']['seed'])
        validation_df, test_df = train_test_split(df0,
                                                  train_size = float(args['preprocess_data']['validation_share'])/(1 - float(args['preprocess_data']['training_share'])),
                                                  stratify = df0[args['data_labels']['labels_column']],
                                                  random_state = args['preprocess_data']['seed'])

        df_d = {0: train_df, 1: validation_df, 2: test_df}

        if args['preprocess_data']['rebalance_data']:
            sm = SMOTE('not majority')
            for k in df_d:
                df_d[k].reset_index(drop = True, inplace = True)
                temp_index, _ = sm.fit_resample(df_d[k].index.to_numpy().reshape(-1, 1), pd.Categorical(df_d[k][args['data_labels']['labels_column']]).codes)
                df_d[k] = pd.DataFrame(temp_index).set_index(0).merge(df_d[k], how = 'left', left_index = True, right_index = True, validate = 'm:1')

        if args['preprocess_data']['reweigh_data']:
            for k, v in df_d.items():
                df_d[k].reset_index(drop = True, inplace = True)
                df_d[k] = pd.DataFrame(np.random.choice(df_d[k].index, size = int(len(df_d[k]) * args['preprocess_data']['reweigh_factor']), p = df_d[k][args['preprocess_data']['weights_column']] / df_d[k][args['preprocess_data']['weights_column']].sum())).set_index(0).merge(df_d[k], how = 'left', left_index = True, right_index = True,  validate = 'm:1')

        for x, name, header in [(pd.Series(args['data_labels']['labels_list']), 'labels.csv', False),
                                (df_d[0], 'train.csv', True),
                                (df_d[1], 'validation.csv', True),
                                (df_d[2], 'test.csv', True)]:
            x.to_csv(data_folder / name, header = header, index = False)

        del(df0, x, name, header)

    ### Data processing:
    #logging.info('Start {}: {}'.format(job_label, pd.Timestamp.today(tz = 'UTC')))
    tokenizer = BertTokenizer.from_pretrained(args['BERT_model'], do_lower_case = args['lower_case'])

    databunch = BertDataBunch(data_folder, data_folder, tokenizer = tokenizer,
                              train_file = 'train.csv', val_file = 'validation.csv', test_data = 'test.csv', label_file = 'labels.csv',
                              text_col = args['data_labels']['text'], label_col = args['data_labels']['labels_column'],
                              bs = args['training_batch_size'],
                              maxlen = args['max_sequence_length'],
                              multi_gpu = multi_gpu,
                              multi_label = False)

    ### Metrics setup:
    metrics = []
    for k, v in args['metrics']['functions'].items():
        metrics.append({'name': k, 'function': v})

    ### Model training:
    learner = BertLearner.from_pretrained_model(databunch, args['BERT_model'], metrics, device, logger,
                                                finetuned_wgts_path = None, # FINETUNED_PATH,
                                                warmup_proportion = args['warmup_proportion'],
                                                grad_accumulation_steps = args['gradient_accumulation_steps'],
                                                multi_gpu = multi_gpu,
                                                is_fp16 = args['FP16'],
                                                loss_scale = args['FP16_loss_scale'],
                                                multi_label = False)

    learner.fit(args['training_epochs'], lr = args['learning_rate'], schedule_type = args['warmup_schedule'])

    learner.save_and_reload(args['folders']['output'], '{}-{}'.format(args['job_label'], run_start_time).replace(' ', ''))

## Run definitions
negativereason_labels = list(df0.negativereason.value_counts().index)
run_definitions = {
        'tweets_BERT_sentiment_01_basic':
            {
            'labels': list(palette.keys()),
            'rebalance_data': False,
            'reweigh_data': False,
            'weights_column': None,
            'input_file': train_full_file,
            'labels_column': 'airline_sentiment',
            },
        'tweets_BERT_sentiment_02_rebalance':
            {
            'labels': list(palette.keys()),
            'rebalance_data': True,
            'reweigh_data': False,
            'weights_column': None,
            'input_file': train_full_file,
            'labels_column': 'airline_sentiment',
            },
        'tweets_BERT_sentiment_03_reweigh':
            {
            'labels': list(palette.keys()),
            'rebalance_data': False,
            'reweigh_data': True,
            'weights_column': 'airline_sentiment_confidence',
            'input_file': train_full_file,
            'labels_column': 'airline_sentiment',
            },
        'tweets_BERT_sentiment_04_rebalance_reweigh':
            {
            'labels': list(palette.keys()),
            'rebalance_data': True,
            'reweigh_data': True,
            'weights_column': 'airline_sentiment_confidence',
            'input_file': train_full_file,
            'labels_column': 'airline_sentiment',
            },
        'tweets_BERT_negatives_01_basic':
            {
            'labels': negativereason_labels,
            'rebalance_data': False,
            'reweigh_data': False,
            'weights_column': None,
            'input_file': train_negatives_file,
            'labels_column': 'negativereason',
            },
        'tweets_BERT_negatives_02_rebalance':
            {
            'labels': negativereason_labels,
            'rebalance_data': True,
            'reweigh_data': False,
            'weights_column': None,
            'input_file': train_negatives_file,
            'labels_column': 'negativereason',
            },
        'tweets_BERT_negatives_03_reweigh':
            {
            'labels': negativereason_labels,
            'rebalance_data': False,
            'reweigh_data': True,
            'weights_column': 'negativereason_confidence',
            'input_file': train_negatives_file,
            'labels_column': 'negativereason',
            },
        'tweets_BERT_negatives_04_rebalance_reweigh':
            {
            'labels': negativereason_labels,
            'rebalance_data': True,
            'reweigh_data': True,
            'weights_column': 'negativereason_confidence',
            'input_file': train_negatives_file,
            'labels_column': 'negativereason',
            }
        }

### **Model training execution**
for k, v in run_definitions.items():
    train_model(job_label = k,
        labels_list = v['labels'],
        rebalance_data = v['rebalance_data'],
        reweigh_data = v['reweigh_data'],
        weights_column = v['weights_column'],
        input_file = v['input_file'],
        labels_column = v['labels_column']
        )
