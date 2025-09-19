import os
import whisper
import yt_dlp
import tempfile
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api.formatters import TextFormatter
from crewai import Agent, Task, Crew
import streamlit as st
import re
import pandas as pd

# -------------------- Display Fact-Check Output --------------------
def display_fact_check_output(raw_output, show_table=True):
    """
    Display CrewAI fact-checking results as markdown and optional table.
    Expected format from Reporter agent:
    1. Claim: "..."
       - Status: True/False
       - Reasoning: ...
       - Context: ...
       - References: ...
    """
    st.markdown("### ✅ Fact-Check Results")

    # Split by numbered claims
    claims = re.split(r"\n(?=\d+\.\s*Claim:)", raw_output.strip())
    claims_data = []

    for claim in claims:
        lines = [l.strip() for l in claim.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        claim_text = next((l.split("Claim:")[1].strip().strip('"') for l in lines if "Claim:" in l), "")
        status_text = next((l.split("Status:")[1].strip() for l in lines if "Status:" in l), "Unknown")
        reasoning_text = next((l.split("Reasoning:")[1].strip() for l in lines if "Reasoning:" in l), "")
        context_text = next((l.split("Context:")[1].strip() for l in lines if "Context:" in l), "")
        references_text = next((l.split("References:")[1].strip() for l in lines if "References:" in l), "")

        claims_data.append({
            "Claim": claim_text,
            "Status": status_text,
            "Reasoning": reasoning_text,
            "Context": context_text,
            "References": references_text
        })

        # Display in markdown
        st.markdown(f"- **Claim**: {claim_text}")
        st.markdown(f"  - **Status**: `{status_text}`")
        if reasoning_text:
            st.markdown(f"  - **Reasoning**: {reasoning_text}")
        if context_text:
            st.markdown(f"  - **Context**: {context_text}")
        if references_text:
            st.markdown(f"  - **References**: {references_text}")

    if show_table and claims_data:
        st.markdown("---")
        st.markdown("### 📊 Fact-Check Summary Table")
        df = pd.DataFrame(claims_data)
        st.dataframe(df, use_container_width=True)


# -------------------- Environment Setup --------------------
os.environ["OPENAI_API_KEY"] = "your_openai_key"  # 
FFMPEG_PATH = r"file_path_to_ffmpeg"  # Update with your ffmpeg path
os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# -------------------- Helper Functions --------------------
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    return None


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except (NoTranscriptFound, TranscriptsDisabled):
        raise RuntimeError("No transcript available for this video.")
    except VideoUnavailable:
        raise RuntimeError("Video is unavailable or private.")
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")


def download_audio(youtube_url):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': FFMPEG_PATH,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    for file in os.listdir(temp_dir):
        if file.endswith(".mp3"):
            return os.path.join(temp_dir, file)
    raise FileNotFoundError("Audio file not found.")


def generate_transcript_from_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']


# -------------------- CrewAI Multi-Agent Workflow --------------------
def fact_check_with_crewai(transcript_text):
    # Define agents
    summarizer = Agent(
        role="Summarizer",
        goal="Summarize transcript into key factual claims",
        backstory="An expert at condensing long transcripts into concise factual claims.",
        allow_delegation=False,
        verbose=True
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal="Verify the accuracy of claims",
        backstory="A domain expert who validates claims against reliable sources.",
        allow_delegation=False,
        verbose=True
    )

    context_provider = Agent(
        role="Context Provider",
        goal="Provide context and references for each claim",
        backstory="A research assistant who provides supporting context and sources.",
        allow_delegation=False,
        verbose=True
    )

    reporter = Agent(
        role="Reporter",
        goal="Format results clearly in markdown with claims, statuses, reasoning, context, and references",
        backstory="A skilled writer who creates structured and easy-to-read reports.",
        allow_delegation=False,
        verbose=True
    )

    # Define tasks
    task1 = Task(
        description=f"Summarize this transcript into key factual claims:\n\n{transcript_text[:3000]}",
        expected_output="A numbered list of factual claims.",
        agent=summarizer
    )

    task2 = Task(
        description="Fact-check each claim and mark as true or false with reasoning.",
        expected_output="List of claims with verification status and reasoning.",
        agent=fact_checker,
        depends_on=[task1]
    )

    task3 = Task(
        description="Provide context and references for each claim.",
        expected_output="Claims with additional background and references.",
        agent=context_provider,
        depends_on=[task2]
    )

    task4 = Task(
        description="Format the fact-check results into clean markdown with Claim, Status, Reasoning, Context, References.",
        expected_output="Structured markdown report.",
        agent=reporter,
        depends_on=[task3]
    )

    # Run crew
    crew = Crew(agents=[summarizer, fact_checker, context_provider, reporter],
                tasks=[task1, task2, task3, task4],
                verbose=True)
    return crew.kickoff()


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="YouTube Fact Checker", layout="centered")
st.title("🎥 YouTube Video Fact Checker with CrewAI")

youtube_url = st.text_input("Enter a YouTube Video URL:")
start_check = st.button("🔍 Analyze & Fact Check")

if start_check and youtube_url:
    try:
        # ✅ Extract video ID
        with st.spinner("Extracting video ID..."):
            video_id = get_video_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL.")
                st.stop()

        # ✅ Try fetching transcript from YouTube
        try:
            with st.spinner("Fetching transcript from YouTube..."):
                transcript = get_transcript(video_id)

        except RuntimeError as e:
            st.warning(f"{e}\nUsing Whisper for audio transcription...")
            with st.spinner("Downloading audio..."):
                audio_path = download_audio(youtube_url)
            with st.spinner("Transcribing audio with Whisper..."):
                transcript = generate_transcript_from_audio(audio_path)

        # ✅ Run fact-check
        with st.spinner("🧠 Running Fact Check using CrewAI..."):
            result = fact_check_with_crewai(transcript)

        st.success("✅ Fact Check Completed")

        # ✅ Debug: Show raw CrewAI output
        st.subheader("🔍 Debug: Raw CrewAI Output")
        st.write(result)

        # ✅ Show transcript
        st.subheader("📜 Transcript Preview")
        st.text_area(
            "Transcript",
            transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
            height=300
        )

        # ✅ Show fact-check result
        st.subheader("🧾 Fact-Check Result")

        if not result:
            st.error("❌ CrewAI returned nothing.")
        elif isinstance(result, str):
            display_fact_check_output(result)
        elif isinstance(result, list) and len(result) > 0:
            final_output = getattr(result[-1], "output", str(result[-1]))
            display_fact_check_output(final_output)
        elif hasattr(result, "output"):
            display_fact_check_output(result.output)
        else:
            display_fact_check_output(str(result))

    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
