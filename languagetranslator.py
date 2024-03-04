import streamlit as st
import time
import keyboard
import os
import psutil

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Define language options and their corresponding codes
lang_options = ['Arabic','Danish', 'English', 'French', 'German', 'Italian', 'Russian', 'Spanish', 'Swedish', 'Zhongwen(Chinese)']
lang_tuple = (("Arabic", "ar"), ("Danish","da"), ("English","en"), ("French","fr"), ("German","de"),
               ("Italian","it"), ("Russian","ru"), ("Spanish", "es"), ("Swedish","sv"), ("Zhongwein","zh"))

src_lang_text = 'Select source language:'
dest_lang_text = 'Select target language:'

def languageselect(src_dest):
    """
    Function to create a selectbox for language selection.

    Parameters:
    - src_dest (str): Text indicating whether it's source or destination language selection.

    Returns:
    - str: Language code selected by the user.
    """
    sel_lang_name = st.selectbox(src_dest, lang_options)
    index = lang_options.index(sel_lang_name) 
    return lang_tuple[index][1]

def translate(text, source_language, target_language):
    """
    Function to translate text from source language to target language using the Transformers pipeline.

    Parameters:

        - text (str): Input text to be translated.
    - source_language (str): Source language code.
    - target_language (str): Target language code.

    Returns:
    - str: Translated text.
    """
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{source_language}-{target_language}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{source_language}-{target_language}")

    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    translation = translator(text, source_language, target_language)
    translated_text = translation[0]['translation_text']
    return translated_text

def main():
    """
    Main function to create the Streamlit app for language translation.
    """
    st.set_page_config(layout='wide')
    st.header('Trivikram Prasad :dog:', divider='rainbow')
    st.title('Language Translator using AI')
    source_text = st.text_area("Enter the text to translate:")

    # Select source and target languages
    source_language = languageselect(src_lang_text)
    target_language = languageselect(dest_lang_text) 

    if st.button("Translate"):
        # Translate text and display result
        translated_text = translate(source_text, source_language, target_language)
        st.write("Translated Text:", translated_text)

    # Button to shut down the app
    exit_app = st.button("Quit Translator")
    if exit_app:
        # Give a bit of delay for user experience
        time.sleep(5)
        # Close streamlit browser tab
        #keyboard.press_and_release('ctrl+w') (needs root access)
        # Terminate streamlit python process
        pid = os.getpid()
        p = psutil.Process(pid)
        p.terminate()


if __name__== "__main__":
    main()


