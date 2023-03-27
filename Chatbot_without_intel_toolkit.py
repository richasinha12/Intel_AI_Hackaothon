#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## for language model
import transformers

## for data
#import os
import datetime
import numpy as np


# In[ ]:


import os


# In[ ]:


import pyttsx3


# In[ ]:


import speech_recognition as sr


# In[ ]:


import time


# In[ ]:


# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name
        
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
             recognizer.adjust_for_ambient_noise(mic, duration=1)            
             print("listening...")
             audio = recognizer.listen(mic)
        try:
             self.text = recognizer.recognize_google(audio)
             print("me --> ", self.text)
        except:
             print("me -->  ERROR")
    
    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def wake_up(self, text):
        return True if self.name in text.lower() else False
    
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')
    
   

        # Run the AI
if __name__ == "__main__":
    ai = ChatBot(name="maya")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        ai.speech_to_text()

            ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Maya the AI, what can I do for you?"

            ## action time
        elif "time" in ai.text:
            res = ai.action_time()

            ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            # get the start time
            st = time.time()
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])
            # get the end time
            et = time.time()
            # get the execution time
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds')
            ## conversation
        else:   
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




