import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError)
from PIL import Image
import datetime
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

#Set up VietOCR
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False

detector = Predictor(config)

#List of info area
l = [[170, 106, 330, 128],      # Work permit number
     [120, 170, 330, 192],      # Name
     [95, 210, 130, 230],       # Gender
     [120, 245, 330, 265],      # DOB
     [110, 283, 205, 303],      # Nationality
     [260, 283, 330, 303],      # Passport number
     [168, 320, 342, 339],      # Company name 1
     [168, 337, 342, 353],      # Company name 2
     [109, 359, 342, 379],      # Company address 1
     [108, 377, 342, 391],      # Company address 2
     [465, 42, 670, 62],        # Position
     [410, 127, 435, 142],      # Working from day
     [458, 127, 483, 142],      # Working from month
     [500, 127, 525, 142],      # Working from year
     [562, 127, 587, 142],      # Working to day
     [613, 127, 633, 142],      # Working to month
     [650, 127, 680, 142]]      # Working to year

df_data = {'executed_date': [],
        'file_name': [],
        # 'image': [],
        'doc_number': [],
        'name': [],
        'gender': [],
        'dob': [],
        'nationality': [],
        'passport_num': [],
        'company_name': [],
        'company_add': [],
        'position': [],
        'working_from': [],
        'working_to': []}

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

menu = ['Upload file']

choice = st.sidebar.selectbox('Upload file', menu)

if 'img' not in st.session_state:
    st.session_state['img'] = 0
if 'name' not in st.session_state:
    st.session_state['name'] = 0

if choice == 'Upload file':
    file_upload = st.file_uploader('Upload file', type=['jpeg','jpg','png','pdf'])
    if file_upload != None:
        if file_upload.name[-3:] == 'pdf':
            images = convert_from_bytes(file_upload.read(), fmt='PNG', size=(700,500))
            img_array = np.array(images[0])
            st.image(img_array)
        else:
            images = Image.open(file_upload)
            img_array = np.array(images)
            img_array = cv2.resize(img_array, dsize=(700,500))
            st.image(img_array)
        
        cr = []
        for i in range(len(l)):
            cr.append(img_array[l[i][1]:l[i][3], l[i][0]:l[i][2]])
        
        scale = 2.5
        result_test = []
        for i in range(len(cr)):
            if i == 3 or i == 8:
                img = cv2.resize(cr[i], dsize=(int(cr[i].shape[1]*scale),int(cr[i].shape[0]*scale)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
                img = Image.fromarray(img, 'L')
                result_test.append(detector.predict(img, return_prob=False))
            elif i == 4:
                img = cv2.resize(cr[i], dsize=(int(cr[i].shape[1]*scale),int(cr[i].shape[0]*scale)))
                img = Image.fromarray(img, 'RGB')
                result_test.append(detector.predict(img, return_prob=False))
            else:
                img = Image.fromarray(cr[i], 'RGB')
                result_test.append(detector.predict(img, return_prob=False))
        
        remove_chars = [',', '.','/']

        for character in remove_chars:
            result_test[2] = result_test[2].replace(character, ' ')
            result_test[3] = result_test[3].replace(character, '')
            result_test[4] = result_test[4].replace(character, ' ')
            result_test[5] = result_test[5].replace(character, ' ')
            result_test[6] = result_test[6].replace(character, ' ')
            result_test[10] = result_test[10].replace(character, ' ')

        # st.write(result_test)
        st.write('Execution date:', datetime.date.today())
        st.write('File name:', file_upload.name)
        st.write('Documentation number:', result_test[0][:(len(result_test[0])-len(result_test[0][6:]))]+'-'+result_test[0][6:])
        st.write('Full name:', result_test[1])
        st.write('Gender:', 'Nam' if result_test[2] != None else 'Nu')
        
        try:
            st.write('Day of birth:', datetime.date(int(result_test[3][4:]), int(result_test[3][2:4]), int(result_test[3][:2])))
        except: 
            result_test[3] = 'Cannot retrieve data'
            st.write('Day of birth:', result_test[3])     
        
        st.write('Nationality:', result_test[4])
        st.write('Passport number:', result_test[5])
        st.write('Comapany name:', result_test[6]+' '+result_test[7])
        st.write('Company address:', result_test[8]+' '+result_test[9])
        st.write('Position:', result_test[10])

        try:
            st.write('Working from:', datetime.date(int(result_test[13]), int(result_test[12]), int(result_test[11])))
        except:
            st.write('Working from: Cannot retrieve data')
        
        try:
            st.write('Working to:', datetime.date(int(result_test[16]), int(result_test[15]), int(result_test[14])))
        except: 
            st.write('Working to: Cannot retrieve data')
        
        df_data['executed_date'].append(datetime.date.today())
        df_data['file_name'].append(file_upload.name)
        df_data['doc_number'].append(result_test[0][:(len(result_test[0])-len(result_test[0][6:]))]+'-'+result_test[0][6:])
        df_data['name'].append(result_test[1])
        df_data['gender'].append('Nam' if result_test[2] == 'x' or result_test[2] == 'X' else 'Nu')
        try:
            df_data['dob'].append(datetime.date(int(result_test[3][4:]), int(result_test[3][2:4]), int(result_test[3][:2])))
        except:
            df_data['dob'].append('Cannot retrieve data')        
        df_data['nationality'].append(result_test[4])
        df_data['passport_num'].append(result_test[5])
        df_data['company_name'].append(result_test[6]+' '+result_test[7])
        df_data['company_add'].append(result_test[8]+' '+result_test[9])
        df_data['position'].append(result_test[10])
        try:
            df_data['working_from'].append(datetime.date(int(result_test[13]), int(result_test[12]), int(result_test[11])))
        except:
            df_data['working_from'].append('Cannot retrieve data')

        try:
            df_data['working_to'].append(datetime.date(int(result_test[16]), int(result_test[15]), int(result_test[14])))
        except: 
            df_data['working_to'].append('Cannot retrieve data')

        st.write('Data is saved to dataframe')

        df = pd.DataFrame.from_dict(df_data)


        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(df)

        st.download_button(
        'Extract to csv',
        csv,
        "data.csv",
        "text/csv",
        key='download-csv')