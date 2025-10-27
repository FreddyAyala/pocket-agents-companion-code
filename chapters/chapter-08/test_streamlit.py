import streamlit as st
import time

st.title('🤖 On-Device AI Assistant')
st.write('This is a simple Streamlit app!')

if st.button('Test AI'):
    with st.spinner('Processing...'):
        time.sleep(1)
    st.success('AI processed your request locally!')

user_input = st.text_input('Enter your message:')
if user_input:
    st.write(f'You said: {user_input}')
    st.write('AI Response: Message processed locally on your device!')

st.markdown('## ✅ Streamlit is working!')
st.info('This app is running locally on your device.')
