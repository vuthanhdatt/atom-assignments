import streamlit as st
import json
import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime as dt
from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib

st.set_page_config(layout="centered",
                   page_title="DataCracy Dashboard",
                   page_icon = '💙')


#SLACK_BEARER_TOKEN = os.environ.get('SLACK_BEARER_TOKEN') ## Get in setting of Streamlit Share
SLACK_BEARER_TOKEN = st.secrets["TOKEN"]
DTC_GROUPS_URL = ('https://raw.githubusercontent.com/anhdanggit/atom-assignments/main/data/datacracy_groups.csv')
#st.write(json_data['SLACK_BEARER_TOKEN'])

@st.cache
def load_users_df():
    # Slack API User Data
    endpoint = "https://slack.com/api/users.list"
    headers = {"Authorization": "Bearer {}".format(SLACK_BEARER_TOKEN)}
    response_json = requests.post(endpoint, headers=headers).json() 
    user_dat = response_json['members']

    # Convert to CSV
    user_dict = {'user_id':[],'name':[],'display_name':[],'real_name':[],'title':[],'is_bot':[]}
    for i in range(len(user_dat)):
      user_dict['user_id'].append(user_dat[i]['id'])
      user_dict['name'].append(user_dat[i]['name'])
      user_dict['display_name'].append(user_dat[i]['profile']['display_name'])
      user_dict['real_name'].append(user_dat[i]['profile']['real_name_normalized'])
      user_dict['title'].append(user_dat[i]['profile']['title'])
      user_dict['is_bot'].append(int(user_dat[i]['is_bot']))
    user_df = pd.DataFrame(user_dict) 
    # Read dtc_group hosted in github
    dtc_groups = pd.read_csv(DTC_GROUPS_URL)
    user_df = user_df.merge(dtc_groups, how='left', on='name')
    return user_df

@st.cache
def load_channel_df():
    endpoint2 = "https://slack.com/api/conversations.list"
    data = {'types': 'public_channel,private_channel'} # -> CHECK: API Docs https://api.slack.com/methods/conversations.list/test
    headers = {"Authorization": "Bearer {}".format(SLACK_BEARER_TOKEN)}
    response_json = requests.post(endpoint2, headers=headers, data=data).json() 
    channel_dat = response_json['channels']
    channel_dict = {'channel_id':[], 'channel_name':[], 'is_channel':[],'creator':[],'created_at':[],'topics':[],'purpose':[],'num_members':[]}
    for i in range(len(channel_dat)):
        channel_dict['channel_id'].append(channel_dat[i]['id'])
        channel_dict['channel_name'].append(channel_dat[i]['name'])
        channel_dict['is_channel'].append(channel_dat[i]['is_channel'])
        channel_dict['creator'].append(channel_dat[i]['creator'])
        channel_dict['created_at'].append(dt.fromtimestamp(float(channel_dat[i]['created'])))
        channel_dict['topics'].append(channel_dat[i]['topic']['value'])
        channel_dict['purpose'].append(channel_dat[i]['purpose']['value'])
        channel_dict['num_members'].append(channel_dat[i]['num_members'])
    channel_df = pd.DataFrame(channel_dict) 
    return channel_df

@st.cache(allow_output_mutation=True)
def load_msg_dict():
    endpoint3 = "https://slack.com/api/conversations.history"
    headers = {"Authorization": "Bearer {}".format(SLACK_BEARER_TOKEN)}
    msg_dict = {'channel_id':[],'msg_id':[], 'msg_ts':[], 'user_id':[], 'latest_reply':[],'reply_user_count':[],'reply_users':[],'github_link':[],'text':[]}
    for channel_id, channel_name in zip(channel_df['channel_id'], channel_df['channel_name']):
        print('Channel ID: {} - Channel Name: {}'.format(channel_id, channel_name))
        try:
            data = {"channel": channel_id} 
            response_json = requests.post(endpoint3, data=data, headers=headers).json()
            msg_ls = response_json['messages']
            for i in range(len(msg_ls)):
                if 'client_msg_id' in msg_ls[i].keys():
                    msg_dict['channel_id'].append(channel_id)
                    msg_dict['msg_id'].append(msg_ls[i]['client_msg_id'])
                    msg_dict['msg_ts'].append(dt.fromtimestamp(float(msg_ls[i]['ts'])))
                    msg_dict['latest_reply'].append(dt.fromtimestamp(float(msg_ls[i]['latest_reply'] if 'latest_reply' in msg_ls[i].keys() else 0))) ## -> No reply: 1970-01-01
                    msg_dict['user_id'].append(msg_ls[i]['user'])
                    msg_dict['reply_user_count'].append(msg_ls[i]['reply_users_count'] if 'reply_users_count' in msg_ls[i].keys() else 0)
                    msg_dict['reply_users'].append(msg_ls[i]['reply_users'] if 'reply_users' in msg_ls[i].keys() else 0) 
                    msg_dict['text'].append(msg_ls[i]['text'] if 'text' in msg_ls[i].keys() else 0) 
                    ## -> Censor message contains tokens
                    text = msg_ls[i]['text']
                    github_link = re.findall('(?:https?://)?(?:www[.])?github[.]com/[\w-]+/?', text)
                    msg_dict['github_link'].append(github_link[0] if len(github_link) > 0 else None)
        except:
            print('====> '+ str(response_json))
    msg_df = pd.DataFrame(msg_dict)
    return msg_df

def process_msg_data(msg_df, user_df, channel_df):
    ## Extract 2 reply_users
    msg_df['reply_user1'] = msg_df['reply_users'].apply(lambda x: x[0] if x != 0 else '')
    msg_df['reply_user2'] = msg_df['reply_users'].apply(lambda x: x[1] if x != 0 and len(x) > 1 else '')
    ## Merge to have a nice name displayed
    msg_df = msg_df.merge(user_df[['user_id','name','DataCracy_role']].rename(columns={'name':'submit_name'}), \
        how='left',on='user_id')
    msg_df = msg_df.merge(user_df[['user_id','name']].rename(columns={'name':'reply1_name','user_id':'reply1_id'}), \
        how='left', left_on='reply_user1', right_on='reply1_id')
    msg_df = msg_df.merge(user_df[['user_id','name']].rename(columns={'name':'reply2_name','user_id':'reply2_id'}), \
        how='left', left_on='reply_user2', right_on='reply2_id')
    ## Merge for nice channel name
    msg_df = msg_df.merge(channel_df[['channel_id','channel_name','created_at']], how='left',on='channel_id')
    ## Format datetime cols
    msg_df['created_at'] = msg_df['created_at'].dt.strftime('%Y-%m-%d')
    msg_df['msg_date'] = msg_df['msg_ts'].dt.strftime('%Y-%m-%d')
    msg_df['msg_time'] = msg_df['msg_ts'].dt.strftime('%H:%M')
    msg_df['wordcount'] = msg_df.text.apply(lambda s: len(s.split()))
    return msg_df

def convert_date(s):
  if s == 0:
    d = 'Mon'
  elif s == 1:
    d = 'Tue'
  elif s == 2:
    d = 'Wed'
  elif s ==3:
    d = 'Thu'
  elif s == 4:
    d = 'Fri'
  elif s == 5:
    d = 'Sat'
  elif s == 6:
    d = 'Sun' 
  return d
user_df = load_users_df()
channel_df = load_channel_df()
msg_df = load_msg_dict()
msg_df = process_msg_data(msg_df, user_df, channel_df)

submit_df = msg_df[msg_df.channel_name.str.contains('assignment')]
submit_df = submit_df[submit_df.DataCracy_role.str.contains('Learner')]
latest_ts = submit_df.groupby(['channel_name', 'user_id']).msg_ts.idxmax() ## -> Latest ts
submit_df = submit_df.loc[latest_ts]

len_submit = len(submit_df)
len_review = len(submit_df[submit_df['reply_users']!= 0])

submit_gr1_df = submit_df.groupby('DataCracy_role').get_group('Learner_Gr1')

len_submit_1 = len(submit_gr1_df)
len_review_1 = len(submit_gr1_df[submit_gr1_df['reply_users']!= 0])

submit_gr2_df = submit_df.groupby('DataCracy_role').get_group('Learner_Gr2')

len_submit_2 = len(submit_gr2_df)
len_review_2 = len(submit_gr2_df[submit_gr2_df['reply_users']!= 0])

submit_gr3_df = submit_df.groupby('DataCracy_role').get_group('Learner_Gr3')

len_submit_3 = len(submit_gr3_df)
len_review_3 = len(submit_gr3_df[submit_gr3_df['reply_users']!= 0])

submit_gr4_df = submit_df.groupby('DataCracy_role').get_group('Learner_Gr4')

len_submit_4 = len(submit_gr4_df)
len_review_4 = len(submit_gr4_df[submit_gr4_df['reply_users']!= 0])


submit_df['msg_date'] = pd.to_datetime(submit_df['msg_date'])
submit_date = submit_df['msg_date'].dt.weekday
submit_date = submit_date.apply(convert_date)
submit_date = submit_date.value_counts().reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

channel_submit = submit_df.groupby('channel_name').size()
channel_submit = channel_submit.reset_index().rename(columns={0:'count'})
st.title("DataCracy Submit Dashboard")

matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Open Sans"
plt.rcParams['patch.edgecolor'] = 'black'


select = list(pd.unique(submit_df['DataCracy_role']))
select.append('All Group')
option = st.selectbox(
    'Choose submit by group',
    sorted(select))

if option == 'Learner_Gr1':
    st.write(f'Group 1 submited {len_submit_1} assignments')
    st.write(f'Group 1 have {len_review_1} submit be reviewed')
    sizes_all =np.array([len_review, len_submit-len_review])
    sizes_gr1 =np.array([len_review_1, len_submit_1-len_review_1])
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    # sizes =[len_submit, len_submit-len_review]

    pie, (ax1,ax2) = plt.subplots(1,2)
    ax1.pie(sizes_all,autopct=lambda pct: func(pct, sizes_all), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax1.set_title('All group submit', fontdict={'fontsize': 10})

    ax2.pie(sizes_gr1,autopct=lambda pct: func(pct, sizes_gr1), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax2.set_title('Group 1 submit',fontdict={'fontsize': 10})

    pie.legend(['reviewed', 'non reviewed'])


    st.pyplot(pie)
elif option == 'Learner_Gr2':
    st.write(f'Group 2 submited {len_submit_2} assignments')
    st.write(f'Group 2 have {len_review_2} submit be reviewed')
    sizes_all =np.array([len_review, len_submit-len_review])
    sizes_gr2 =np.array([len_review_2, len_submit_2-len_review_2])
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    # sizes =[len_submit, len_submit-len_review]

    pie, (ax1,ax2) = plt.subplots(1,2)
    ax1.pie(sizes_all,autopct=lambda pct: func(pct, sizes_all), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax1.set_title('All group submit', fontdict={'fontsize': 10})

    ax2.pie(sizes_gr2,autopct=lambda pct: func(pct, sizes_gr2), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax2.set_title('Group 2 submit',fontdict={'fontsize': 10})

    pie.legend(['reviewed', 'non reviewed'])


    st.pyplot(pie)
elif option == 'Learner_Gr3':
    st.write(f'Group 3 submited {len_submit_3} assignments')
    st.write(f'Group 3 have {len_review_3} submit be reviewed')
    sizes_all =np.array([len_review, len_submit-len_review])
    sizes_gr3 =np.array([len_review_3, len_submit_3-len_review_3])
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    # sizes =[len_submit, len_submit-len_review]

    pie, (ax1,ax2) = plt.subplots(1,2)
    ax1.pie(sizes_all,autopct=lambda pct: func(pct, sizes_all), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax1.set_title('All group submit', fontdict={'fontsize': 10})

    ax2.pie(sizes_gr3,autopct=lambda pct: func(pct, sizes_gr3), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax2.set_title('Group 2 submit',fontdict={'fontsize': 10})

    pie.legend(['reviewed', 'non reviewed'])


    st.pyplot(pie)
elif option == 'Learner_Gr4':
    st.write(f'Group 4 submit {len_submit_4} assignments')
    st.write(f'Group 4 have {len_review_4} submit be reviewed')
    sizes_all =np.array([len_review, len_submit-len_review])
    sizes_gr4 =np.array([len_review_4, len_submit_4-len_review_4])
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)
    # sizes =[len_submit, len_submit-len_review]

    pie, (ax1,ax2) = plt.subplots(1,2)
    ax1.pie(sizes_all,autopct=lambda pct: func(pct, sizes_all), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax1.set_title('All group submit', fontdict={'fontsize': 10})

    ax2.pie(sizes_gr4,autopct=lambda pct: func(pct, sizes_gr4), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax2.set_title('Group 2 submit',fontdict={'fontsize': 10})

    pie.legend(['reviewed', 'non reviewed'])


    st.pyplot(pie)
elif option == 'All Group':
    st.write(f'All learners submited {len_submit} assignments')
    st.write(f'Have {len_review} submit be reviewed ')
    sizes_all =np.array([len_review, len_submit-len_review])
    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d} )".format(pct, absolute)
    # sizes =[len_submit, len_submit-len_review]

    pie, ax1 = plt.subplots()
    ax1.pie(sizes_all,autopct=lambda pct: func(pct, sizes_all), colors=['#ffffff','#e6e6e6'],textprops={'fontsize': 7})
    ax1.set_title('All group submit', fontdict={'fontsize': 10})
    
    pie.legend(['reviewed', 'non reviewed'])

    st.pyplot(pie)   

period = st.slider(
    'Select a range of hours submit',
    0,23,(0,23))
period = list(period)
time_submit = submit_df['msg_time'].apply(lambda x: x.split(':')[0]).value_counts().sort_index()[period[0]:period[1]+1]

sns.set_style("ticks")
bar = plt.figure()
sns.barplot(time_submit.index,time_submit.values,facecolor=(1, 1, 1, 0),edgecolor=".2")
plt.title('Time submit distribution',fontsize=15)
plt.xlabel('Hour')
plt.ylabel('Submit')
st.pyplot(bar)

option1 = st.selectbox(
    'Submit by',
    ['Week','Weekday'])
if option1 == 'Week':
    submit_channel = plt.figure()
    ax1 = sns.barplot(channel_submit.index+1,channel_submit['count'],facecolor=(1, 1, 1, 0),edgecolor=".2")
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=10,
                   xytext = (0, -12), 
                   textcoords = 'offset points')
    plt.title('Submit distribution by week',fontsize=15)
    plt.xlabel('Assignment week')
    plt.ylabel('Submit')
    st.pyplot(submit_channel)
elif option1 == 'Weekday':
    bar_date = plt.figure()

    ax = sns.barplot(submit_date.index,submit_date.values,facecolor=(1, 1, 1, 0),edgecolor=".2", order= submit_date.index)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    size=10,
                    xytext = (0, -12), 
                    textcoords = 'offset points')
    plt.title('Submit distribution by weekday',fontsize=15)
    plt.xlabel('Days of week')
    plt.ylabel('Submit')
    st.pyplot(bar_date)


