import os
import sys
from dotenv import load_dotenv  # type: ignore

load_dotenv()

import google.generativeai as genai  # type: ignore
import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore

api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY environment variable not set. Gemini features will be disabled until set.")

system_instruction = """
You are an expert, multi-platinum music producer, songwriter, and creative co-writer.
Your goal is to assist musicians in the creative process.

Specialties:
1. Lyric Generation & Refinement: Suggesting rhymes, metaphors, and structural improvements (verse, chorus, bridge). Provide context on why a certain line works rhythmically.
2. Chord Progressions: Recommending culturally relevant and genre-specific chord progressions. Provide Roman numerals alongside standard chords.
3. Production Advice: Giving specific tips on instrumentation, arrangement, mixing, and sound design.

Guidelines:
- Always be encouraging, collaborative, and speak like a peer in the studio.
- Provide actionable, specific musical advice rather than generic platitudes.
- Format chord progressions cleanly so they are easy to read.
- If suggesting lyrics, offer a few different thematic variations or directions.
"""

spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

sp = None
if spotify_client_id and spotify_client_secret:
    auth_manager = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    print("Spotify API integration enabled.")
else:
    print("Warning: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set. Spotify features will be disabled.")


def get_track_info(track_name: str, artist_name: str = "") -> dict:
    """
    Searches for a track on Spotify and returns its audio features.

    Args:
        track_name: The name of the song to search for.
        artist_name: Optional name of the artist for a more accurate search.

    Returns:
        dict: A dictionary containing the track name, artist name, tempo (BPM),
              key (musical note), mode (Major/Minor), danceability, and energy.
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

        features = sp.audio_features(track_id)

        if not features or not features[0]:
            return {"error": f"Audio features not available for '{actual_track_name}'."}

        feature = features[0]

        key_mapping = {
            0: 'C', 1: 'C#/Db', 2: 'D', 3: 'D#/Eb', 4: 'E', 5: 'F',
            6: 'F#/Gb', 7: 'G', 8: 'G#/Ab', 9: 'A', 10: 'A#/Bb', 11: 'B'
        }

        key_note = key_mapping.get(feature['key'], 'Unknown')
        mode = "Major" if feature['mode'] == 1 else "Minor"

        if key_note != 'Unknown':
            musical_key = f"{key_note} {mode}"
        else:
            musical_key = "Unknown"

        return {
            "track_name": actual_track_name,
            "artist_name": actual_artist_name,
            "tempo_bpm": round(feature['tempo']),
            "key": musical_key,
            "danceability": feature['danceability'],
            "energy": feature['energy'],
            "time_signature": feature['time_signature']
        }
    except Exception as e:
        return {"error": f"Spotify API error: {str(e)}"}


try:
    tools = [get_track_info] if sp else None
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction,
        tools=tools,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
        )
    )
except Exception as e:
    print(f"Failed to initialize the model: {e}")
    sys.exit(1)


def chat_loop():
    print("Creative Co-writer initialized! (Type 'quit' or 'exit' to stop)")
    print("-" * 60)
    print("Try asking: 'Give me a melancholic, synth-pop chord progression' OR")
    print("'Help me rewrite this second verse to be more punchy.'")
    print("-" * 60)

    chat = model.start_chat(enable_automatic_function_calling=True)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Session ended. Keep making great music!")
                break

            if not user_input.strip():
                continue

            print("Co-writer is thinking...")
            response = chat.send_message(user_input)

            print(f"\nCo-writer:\n{response.text}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession ended. Keep making great music!")
            break
        except Exception as e:
            print(f"\nAn error occurred communicating with Gemini: {e}")


if __name__ == "__main__":
    chat_loop()
