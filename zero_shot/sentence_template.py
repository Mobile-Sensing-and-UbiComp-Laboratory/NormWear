sentence_template = {
    # == PRETRAIN ================================================
    'NASA_TLX': {
        'question_template': [
            'What is the NASA Task Load Index in Percentage?',
            'Can you provide the NASA TLX score as a percentage?',
            'What percentage does the NASA Task Load Index indicate?',
            'Please tell me the NASA Task Load Index in percentage terms.',
            'How much is the NASA TLX score expressed in percentage?'
        ],
        'answer_template':[
            "The NASA Task Load Index (NASA-TLX) was recorded at {}%.",
            "According to the NASA Task Load Index, the workload was {}%.",
            "A score of {}% was measured using NASA-TLX, a common workload assessment tool.",
            "Workload evaluation using NASA TLX resulted in a score of {}%.",
            "The workload score, based on NASA Task Load Index metrics, is {}%.",
            "NASA-TLX, which measures perceived workload across multiple factors, recorded a score of {}%.",
            "An assessment with the NASA Task Load Index showed a workload score of {}%.",
            "Using the NASA-TLX methodology, the workload was rated at {}%.",
            "NASA Task Load Index, a tool for measuring mental and physical workload, indicated {}%.",
            "The recorded workload score from NASA-TLX was {}%.",
            "Workload, assessed via NASA Task Load Index, was measured at {}%.",
            "NASA TLX provided a workload evaluation score of {}%.",
            "The mental and physical demands, as quantified by NASA-TLX, were {}%.",
            "Perceived workload, evaluated using NASA Task Load Index, came out at {}%.",
            "NASA-TLX, a multidimensional workload assessment, reported {}%.",
            "The score derived from NASA Task Load Index data is {}%.",
            "NASA TLX, which evaluates workload across effort, performance, and other factors, reported a score of {}%.",
            "Measured workload, based on NASA-TLX, was {}%.",
            "A workload rating of {}% was determined using NASA Task Load Index methods.",
            "NASA-TLX indicated that the overall workload level was {}%."
        ]
    },
    'PSQI': {
        'question_template': [
            'What is the Pittsburgh Sleep Quality Index in Percentage?',
            'Can you give me the PSQI score as a percentage?',
            'What percentage is the Pittsburgh Sleep Quality Index?',
            'Please provide the PSQI as a percentage.',
            'How is the Pittsburgh Sleep Quality Index represented in percentage?'
        ],
        'answer_template': [
            "The Pittsburgh Sleep Quality Index score is {}%.",
            "According to the PSQI, the score is {}%.",
            "The sleep quality score, measured by the PSQI, stands at {}%.",
            "The PSQI result indicates a score of {}%.",
            "Based on the PSQI, the sleep quality percentage is {}%.",
            "The sleep quality assessment (PSQI) shows a result of {}%.",
            "Measured by the PSQI, the sleep quality index is {}%.",
            "The overall sleep quality, as per the PSQI score, is {}%.",
            "The index for sleep quality, also known as the PSQI, is {}%.",
            "Sleep quality assessment using the PSQI resulted in a score of {}%.",
            "The sleep score (Pittsburgh Sleep Quality Index) is {}%.",
            "Sleep quality percentage, evaluated by PSQI, is {}%.",
            "The reported PSQI sleep quality score stands at {}%.",
            "Sleep quality as indicated by the PSQI is {}%.",
            "The calculated PSQI score for sleep quality is {}%.",
            "Sleep quality, based on the Pittsburgh Sleep Quality Index, is {}%.",
            "The sleep index percentage (PSQI) is {}%.",
            "The PSQI sleep quality evaluation gives a score of {}%.",
            "Based on the Pittsburgh Sleep Quality Index (PSQI), the score is {}%.",
            "Sleep quality evaluation resulted in a score of {}% using the PSQI."
        ]
    },
    'Emotion': {
        'question_template': [
            'What is the current emotion?',
            'Can you identify the current emotion?',
            'What emotion is being expressed now?',
            'What is the emotion right now?',
            'Could you specify the current emotion?'
        ],
        'answer_template': [
            "The emotion recognized is {}.",
            "This person exhibits {} as their emotion.",
            "The detected emotional state is {}.",
            "The emotion associated with this individual is {}.",
            "The current feeling identified is {}.",
            "The observed emotional condition is {}.",
            "This person's emotional experience is {}.",
            "The identified state of emotion is {}.",
            "The analysis suggests the emotion is {}.",
            "Based on observation, the emotion is {}.",
            "The subject's emotional tone appears to be {}.",
            "The prevailing mood detected is {}.",
            "The emotional response identified is {}.",
            "This individual seems to be experiencing {}.",
            "The apparent emotion is {}.",
            "The interpreted emotional state is {}.",
            "The primary feeling of this subject is {}.",
            "The sentiment detected reflects {}.",
            "This individual's emotional signature indicates {}.",
            "The psychological state recognized is {}. Emotion refers to a complex reaction involving physiological and psychological responses to significant stimuli."
        ]
    },
    'Valence': {
        'question_template': [
            'What is the valence level in Percentage?',
            'Can you provide the valence level as a percentage?',
            'What percentage represents the valence level?',
            'Please tell me the valence level in percentage terms.',
            'How much is the valence level expressed in percentage?'
        ],
        'answer_template': [
            'Valence level detected is {}%.',
            'The valence level is {}%.',
            'Valence is measured at {}%.',
            'Measured valence level is {}%.',
            'The detected emotional valence level is {}%.',
            'An emotional valence level of {}% has been identified.',
            'The current valence level reads {}%.',
            'The recorded valence level stands at {}%.',
            'Valence has been assessed at {}%.',
            'Valence detection shows a level of {}%.',
            'The identified valence percentage is {}%.',
            'A valence reading of {}% was obtained.',
            'The emotional tone is measured as {}% valence.',
            'Detection indicates a valence of {}%.',
            'The calculated valence level amounts to {}%.',
            'Assessment reveals a valence level of {}%.',
            'Valence, representing emotional positivity, is {}%.',
            'The positivity score, referred to as valence, is {}%.',
            'An analysis of emotional valence returned a value of {}%.',
            'The percentage for emotional valence detected is {}%.'
        ]
    },
    'Arousal': {
        'question_template': [
            'What is the arousal level in Percentage?',
            'Can you give me the arousal level as a percentage?',
            'What percentage represents the arousal level?',
            'Please provide the arousal level in percentage.',
            'How much is the arousal level expressed in percentage?'
        ],
        'answer_template': [
            "Arousal level detected is {}%.",
            "The arousal level has been recorded as {}%.",
            "The detected arousal level is {}%.",
            "Arousal has been measured at {}%.",
            "The measurement indicates an arousal level of {}%.",
            "The recorded value for arousal is {}%.",
            "Arousal has been quantified at {}%.",
            "The level of arousal has been determined to be {}%.",
            "According to the measurement, the arousal level is {}%.",
            "Based on the analysis, the arousal level is {}%.",
            "The system identifies an arousal level of {}%.",
            "Arousal is reported at {}%.",
            "Arousal has been assessed as {}%.",
            "Arousal intensity is evaluated at {}%.",
            "The arousal percentage recorded is {}%.",
            "The current arousal level stands at {}%.",
            "Measurements show an arousal level of {}%.",
            "Arousal, which reflects the state of alertness or physiological activation, is noted to be {}%.",
            "Physiological arousal, often associated with heightened awareness or stress, has been measured at {}%.",
            "The system's output indicates an arousal metric of {}%."
        ]
    },
    'min_bp': {
        'question_template':[
            'What is the diastolic blood pressure?',
            'Can you provide the diastolic blood pressure reading?',
            'What is the current diastolic BP?',
            'Please tell me the diastolic blood pressure.',
            'What is the diastolic pressure value?'
        ],
        'answer_template': [
            "The diastolic blood pressure was measured as {}.",
            "A diastolic pressure reading of {} was obtained.",
            "Recorded diastolic blood pressure value: {}.",
            "The recorded value for diastolic blood pressure is {}.",
            "Diastolic BP measurement: {}.",
            "A measurement of diastolic BP shows {}.",
            "The diastolic blood pressure was noted to be {}.",
            "The recorded diastolic value is {}.",
            "A diastolic blood pressure reading of {} was documented.",
            "The measured diastolic BP value is {}.",
            "Diastolic blood pressure is the pressure in arteries when the heart is at rest; it was recorded as {}.",
            "The lower number in a blood pressure reading, diastolic pressure, is {}.",
            "A diastolic pressure value of {} has been observed.",
            "Measured diastolic pressure, which represents the arterial pressure during cardiac relaxation, is {}.",
            "Diastolic blood pressure, reflecting the heart's resting state, is {}.",
            "The observed diastolic BP was {}.",
            "A BP reading indicated a diastolic value of {}.",
            "The pressure recorded during heart relaxation, known as diastolic BP, is {}.",
            "The diastolic component of blood pressure is {}.",
            "Blood pressure reading reveals the diastolic value to be {}."
        ]
    },
    'max_bp': {
        'question_template': [
            'What is the systolic blood pressure?',
            'Can you provide the systolic blood pressure reading?',
            'What is the current systolic BP?',
            'Please tell me the systolic blood pressure.',
            'What is the systolic pressure value?'
        ],
        'answer_template': [
            'The systolic blood pressure was recorded as {}.',
            'Systolic blood pressure measurement shows {}.',
            'A systolic blood pressure value of {} was observed.',
            'The recorded systolic pressure is {}.',
            'The systolic blood pressure was measured to be {}.',
            'A measurement of systolic blood pressure indicated a value of {}.',
            'The value for systolic BP is {}.',
            'Systolic blood pressure is noted to be {}.',
            'Systolic blood pressure readings indicate {}.',
            'The systolic reading is {}.',
            'BP, specifically systolic pressure, was recorded as {}.',
            'The top number in the blood pressure reading, known as systolic pressure, is {}.',
            'The systolic blood pressure, which represents the pressure in the arteries when the heart beats, is {}.',
            'The pressure exerted on artery walls during heart contraction, or systolic pressure, is {}.',
            'Blood pressure measurement reports a systolic value of {}.',
            'The systolic component of blood pressure is {}.',
            'Recorded systolic BP value: {}.',
            'The systolic BP reading obtained is {}.',
            'Systolic blood pressure, which measures the force when the heart pumps, is {}.',
            'The initial or higher value in a blood pressure reading, the systolic pressure, measures {}.'
        ]
    },
    'activity': {
        'question_template': [
            'What is the current activity?',
            'Can you identify the current activity?',
            'What activity is happening right now?',
            'Please tell me what the current activity is.',
            'What is the activity being performed currently?'
        ],
        'answer_template':[
            'This subject is presently {}.',
            'The individual is involved in {}.',
            'Currently, the subject is engaged in {}.',
            'This person is performing {} at the moment.',
            'The current task for the subject is {}.',
            'At present, the subject is engaged in {}.',
            'The subject is actively participating in {}.',
            'This individual is involved with {}.',
            'The subject is currently occupied with {}.',
            'The person is presently carrying out {}.',
            'Subject is currently focused on {}.',
            'The subject has taken up {} as their current task.',
            'The ongoing activity of the subject is {}.',
            'The subject is undertaking {} at the moment.',
            'Subject is actively working on {}.',
            'This individual’s present activity is {}.',
            'The individual is currently dedicated to {}.',
            'At this time, the subject is engaged in {}.',
            'The current focus of the subject is {}.',
            'This person is involved in {} right now.'
        ]
    },
    # == DOWNSTREAM ====================================
    ('wearable_downstream/PPG_HTN', 'PPG_HTN'): {
        'type': 'class',
        'question_template': [
            'What is the stage of hypertension?'
        ],
        'answer_template':[
            "The person has no hypertension.",
            "The person is having prehypertension.",
            "The person is at first stage hypertension.",
            "The person is at second stage hypertension.",
        ]
    },
    ('wearable_downstream/PPG_DM', 'PPG_DM'): {
        'type': 'class',
        'question_template': [
            'What is the risk of having diabetes?'
        ],
        'answer_template':[
            "The person is at low risk of having diabetes.",
            "The person is at high risk of having diabetes.",
        ]
    },
    ('wearable_downstream/PPG_CVA', 'PPG_CVA'): {
        'type': 'class',
        'question_template': [
            'What is the risk of having cerebral vascular accident?'
        ],
        'answer_template':[
            "The person is at low risk of having cerebral vascular accident.",
            "The person is at high risk of having cerebral vascular accident."
        ]
    },
    ('wearable_downstream/PPG_CVD', 'PPG_CVD'): {
        'type': 'class',
        'question_template': [
            'What is the risk of having cardiovascular disease?'
        ],
        'answer_template':[
            "The person is at low risk of having cardiovascular disease.",
            "The person is having cerebrovascular disease.",
            "The person is having insufficiency of cerebral blood supply."
        ]
    },
    ('wearable_downstream/indian-fPCG', 'indian-fPCG'): {
        'type': 'reg',
        'question_template': [
            'What is the fetal heart rate?'
        ],
        'answer_template':[
            "The fetal heart rate is {} beats per minute, within the normal range of 110-160 bpm."
        ]
    },
    ('wearable_downstream/ppg_hgb', 'ppg_hgb'): {
        'type': 'reg',
        'question_template': [
            'What is the estimated level of hemoglobin?'
        ],
        'answer_template':[
            "The level of hemoglobin level is {}."
        ]
    },
    ('wearable_downstream/non_invasive_bp', 'non_invasive_bp'): {
        'type': 'reg',
        'question_template': [
            'What is the estimated levels of systolic and diastolic blood pressure?'
        ],
        'answer_template':[
            "The levels of systolic and diastolic blood pressure are {} respectively."
        ]
    },
    ('wearable_downstream/drive_fatigue', 'drive_fatigue'): {
        'type': 'class',
        'question_template': [
            'Is the person in fatigue state or not?'
        ],
        'answer_template':[
            "The person is at normal state.",
            "The person is at fatigue state.",
        ]
    },
    ('wearable_downstream/ecg_heart_cat', 'ecg_heart_cat'): {
        'type': 'class',
        'question_template': [
            'Is the person having myocardial infarction?'
        ],
        'answer_template':[
            "Yes, the person is having high risk of myocardial infarction.",
            "No, the person is having healthy heart beat."
        ]
    },
    ('wearable_downstream/gameemo', 'gameemo'): {
        'type': 'class',
        'question_template': [
            'What is the current emotion state?'
        ],
        'answer_template':[
            "Low mood, low energy.",
            "Low mood, high energy.",
            "High mood, low energy.",
            "High mood, high energy."
        ]
    },
    ('wearable_downstream/uci_har', 'uci_har'): {
        'type': 'class',
        'question_template': [
            'What is the current activity?'
        ],
        'answer_template':[
            "The person is walking.",
            "The person is walking upstairs.",
            "The person is walking downstairs.",
            "The person is sitting.",
            "The person is standing.",
            "The person is laying."
        ]
    },
    ('wearable_downstream/wesad', 'wesad'): {
        'type': 'class',
        'question_template': [
            'Is the person doing good?'
        ],
        'answer_template':[
            "The person is doing so so",
            "The person is under stress.",
            "The person is having fun.",
        ]
    },
    ('wearable_downstream/emg-tfc', 'emg-tfc'): {
        'type': 'class',
        'question_template': [
            'What is the state of the muscle?'
        ],
        'answer_template':[
            "The person has healthy muscle.",
            "The person is suffering from neuropathy.",
            "The person is suffering from myopathy.",
        ]
    },
    ('wearable_downstream/ecg-tfc', 'ecg-tfc'): {
        'type': 'class',
        'question_template': [
            'What is the state of cardiac arrhythmias?'
        ],
        'answer_template':[
            "It is a normal sinus rhythm.",
            "It is atrial fibrillation.",
            "It is alternative rhythm.",
            "Unknown noisy rhythm."
        ]
    },
    ('wearable_downstream/Epilepsy', 'A_Z_eye_open'): {
        'type': 'class',
        'question_template': [
            'What is the status of eyes?'
        ],
        'answer_template':[
            "Eyes closed.",
            "Eyes open.",
        ]
    },
    ('wearable_downstream/Epilepsy', 'B_O_eye_close'): {
        'type': 'class',
        'question_template': [
            'Is the person\'s eyes close? '
        ],
        'answer_template':[
            "No, the eyes are open.",
            "Yes, the eyes are closed.",
        ]
    },
    ('wearable_downstream/Epilepsy', 'C_N_health'): {
        'type': 'class',
        'question_template': [
            'Is this a healthy brain area?'
        ],
        'answer_template':[
            "This might be tumor, or undergo seizure.",
            "This is a normal area."
        ]
    },
    ('wearable_downstream/Epilepsy', 'D_F_tumor'): {
        'type': 'class',
        'question_template': [
            'Is this an area with tumor?'
        ],
        'answer_template':[
            "No, it is not a tumor area.",
            "Yes, it is an area with tumor.",
        ]
    },
    ('wearable_downstream/Epilepsy', 'E_S_seizure'): {
        'type': 'class',
        'question_template': [
            'Is the person experiencing seizure?'
        ],
        'answer_template':[
            "No, the person is not experiencing seizure.",
            "Yes, the person is experiencing seizure.",
        ]
    },
}