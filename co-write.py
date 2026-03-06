import os
import sys
import time
import warnings
from dotenv import load_dotenv  # type: ignore

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai  # type: ignore

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set it in your terminal before running:")
    print('$env:GEMINI_API_KEY="your_api_key_here"')
    sys.exit(1)

genai.configure(api_key=API_KEY)

system_instruction = """
You are an expert, multi-platinum music producer, songwriter, and creative co-writer.
Your goal is to assist musicians in the creative process.

Specialties:
1. Lyric Generation & Refinement: Suggesting rhymes, metaphors, and structural improvements (verse, chorus, bridge). Provide context on why a certain line works rhythmically.
2. Tamil Lyric Generation: You can write Tamil lyrics in romanized English text (Tanglish). When the user asks for Tamil lyrics, write them in English script so they are easy to read and sing. Maintain proper Tamil poetic meter (maathirai) and rhyme schemes used in Tamil film and indie music. Provide an English translation alongside.
3. Chord Progressions: Recommending culturally relevant and genre-specific chord progressions. Provide Roman numerals alongside standard chords.
4. Production Advice: Giving specific tips on instrumentation, arrangement, mixing, and sound design.

Guidelines:
- Always be encouraging, collaborative, and speak like a peer in the studio.
- Provide actionable, specific musical advice rather than generic platitudes.
- Format chord progressions cleanly so they are easy to read.
- If suggesting lyrics, offer a few different thematic variations or directions.
- When writing Tamil lyrics, always use romanized English script (e.g., "Vaanam thirakkum neram" instead of Tamil unicode). Include English meaning in parentheses or as a separate section.
"""

sp = None
spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

if spotify_client_id and spotify_client_secret:
    try:
        import spotipy  # type: ignore
        from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore
        auth_manager = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        print("[OK] Spotify API integration enabled.")
    except Exception as e:
        print(f"[WARN] Could not initialize Spotify: {e}")
else:
    print("[INFO] Spotify keys not set. Spotify features disabled (optional).")


def get_track_info(track_name: str, artist_name: str = "") -> dict:
    """
    Searches for a track on Spotify and returns its details and audio features.

    Args:
        track_name: The name of the song to search for.
        artist_name: Optional name of the artist for a more accurate search.

    Returns:
        dict: A dictionary containing the track name, artist name, album,
              popularity, and audio features (key, tempo, etc.) when available.
              Returns an error dictionary if the track is not found or API fails.
    """
    if not sp:
        return {"error": "Spotify API is not configured. Please set the environment variables."}

    try:
        query = track_name
        if artist_name:
            query += f" artist:{artist_name}"

        results = sp.search(q=query, type='track', limit=1)

        if not results or not results['tracks']['items']:
            return {"error": f"Track '{track_name}' not found on Spotify."}

        track = results['tracks']['items'][0]
        track_id = track['id']
        actual_track_name = track['name']
        actual_artist_name = track['artists'][0]['name']

        track_info = {
            "track_name": actual_track_name,
            "artist_name": actual_artist_name,
            "album": track['album']['name'],
            "release_date": track['album']['release_date'],
            "popularity": track['popularity'],
        }

        try:
            features = sp.audio_features(track_id)
            if features and features[0]:
                f = features[0]
                key_mapping = {
                    0: 'C', 1: 'C#/Db', 2: 'D', 3: 'D#/Eb', 4: 'E', 5: 'F',
                    6: 'F#/Gb', 7: 'G', 8: 'G#/Ab', 9: 'A', 10: 'A#/Bb', 11: 'B'
                }
                key_note = key_mapping.get(f['key'], 'Unknown')
                mode = "Major" if f['mode'] == 1 else "Minor"
                musical_key = f"{key_note} {mode}" if key_note != 'Unknown' else "Unknown"

                track_info["key"] = musical_key
                track_info["tempo_bpm"] = round(f['tempo'])
                track_info["danceability"] = f['danceability']
                track_info["energy"] = f['energy']
                track_info["time_signature"] = f['time_signature']
            else:
                track_info["note"] = "Audio features (key, tempo) not available from Spotify API. Please use your own musical knowledge to provide key, tempo, and other audio characteristics for this track."
        except Exception:
            track_info["note"] = "Audio features (key, tempo) not available from Spotify API. Please use your own musical knowledge to provide key, tempo, and other audio characteristics for this track."

        return track_info
    except Exception as e:
        return {"error": f"Spotify API error: {str(e)}"}


tools = [get_track_info] if sp else None

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        tools=tools,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
        ),
    )
except Exception as e:
    print(f"Failed to initialize the model: {e}")
    sys.exit(1)


def send_with_retry(chat, message, max_retries=3):
    """Send a message to Gemini with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = chat.send_message(message)
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                wait_time = (attempt + 1) * 15
                print(f"  [Rate limited] Waiting {wait_time}s before retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded. The API rate limit is still active. Please wait a minute and try again.")


def chat_loop():
    print()
    print("=" * 60)
    print("  CREATIVE AI CO-WRITER  (Type 'quit' or 'exit' to stop)")
    print("=" * 60)
    print()
    print("Try asking:")
    print("  - 'Give me a melancholic, synth-pop chord progression'")
    print("  - 'Help me rewrite this second verse to be more punchy'")
    if sp:
        print("  - 'What key and tempo is Blinding Lights by The Weeknd?'")
    print()

    chat = model.start_chat(enable_automatic_function_calling=True)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() in ['quit', 'exit']:
                print("\nSession ended. Keep making great music!")
                break

            if not user_input.strip():
                continue

            print("\n  ...thinking...\n")
            response = send_with_retry(chat, user_input)

            print(f"Co-writer:\n{response.text}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nSession ended. Keep making great music!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    chat_loop()
