import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import streamlit as st
import tiktoken


load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_context_window = 195_000


def get_video_docs(video_url: str) -> list[Document]:
    """
     Creates a list of Documents containing video data
     like the video's transcript or YouTube id.
    """

    loader = YoutubeLoader.from_youtube_url(
        video_url, add_video_info=False
    )
    video_docs = loader.load()

    return video_docs


def count_tokens(video_transcript: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(video_transcript))

    return num_tokens


def generate_video_summary(video_url: str, chain_type: str = "stuff") -> str:
    """
    Generates the summary of the video.
    Before the generation, checks the token count and decides which approach to use
    for summary: "map_reduce" if the video transcript token count surpasses
    the used LLM's context window, otherwise it defaults to "stuff".
    """

    video_docs = get_video_docs(video_url)
    video_transcript = video_docs[0].page_content
    num_tokens = count_tokens(video_transcript)

    if num_tokens > llm_context_window:
        chain_type = "map_reduce"

    chain = load_summarize_chain(llm, chain_type=chain_type)
    result = chain.invoke(video_docs)
    video_summary = result["output_text"]

    return video_summary


if __name__ == "__main__":

    st.title("⚡ Summarize YouTube videos ⚡")
    st.subheader("So you can save time for what matters.")

    with st.form("my_form"):
        video_url = st.text_area(
            "Paste the URL of the YouTube video you want summarized:"
        )

        submitted = st.form_submit_button("Get my summary")

        if submitted and video_url.startswith("https://www.youtube.com/watch?v="):
            with st.spinner("Summarizing..."):
                video_summary = generate_video_summary(video_url)

            st.info(video_summary)

        if submitted and not video_url.startswith("https://www.youtube.com/watch?v="):
            st.toast("Please provide a valid YouTube video URL!", icon="😉")