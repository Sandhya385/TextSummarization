import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain.docstore.document import Document


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


api_key=st.sidebar.text_input("Enter the groq api key",type='password')
generic_url=st.text_input("URL",label_visibility='collapsed')
llm=ChatGroq(groq_api_key=api_key,model_name="llama3-8b-8192")

prompt_template=""" Provide the summary of the following content in 300 words:
content={text}"""

prompt=PromptTemplate(input_variables=['text'],template=prompt_template)

def extract_video_id(url):
    """Extracts YouTube video ID."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def fetch_youtube_transcript(video_url):
    """Fetches the YouTube transcript and handles errors."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL format!"

    try:
        st.write(f"🔍 Extracted Video ID: {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript])
        return full_text
    except Exception as e:
        return f"❌ Error fetching transcript: {e}"

if st.button("Summarize the content from YT or URL"):
    #Validate all the inputs
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide the valid information")
    elif not validators.url(generic_url):
        st.error("Please provide the valid url. It can be YT url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript_text = fetch_youtube_transcript(generic_url)
                    if transcript_text.startswith("❌"):
                        st.error(transcript_text)
                        st.stop()  # Stop execution if transcript fails
                    docs = [Document(page_content=transcript_text)]   # Wrap in list for LangChain
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                    docs = loader.load()

                #chain for summarization
                chain=load_summarize_chain(llm,chain_type='stuff',prompt=prompt)
                output_summary=chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

