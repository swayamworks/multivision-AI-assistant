import os
import pandas as pd
from feature_extractor import AudioFeatureExtractor

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "dataset"
OUTPUT_PATH = "processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

extractor = AudioFeatureExtractor()

# =====================================================
# EMOTION MAPPING
# =====================================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

gender_rows = []
emotion_rows = []

print("=" * 60)
print("Starting preprocessing...")
print("=" * 60)

# =====================================================
# LOOP THROUGH DATASET
# =====================================================

for actor_folder in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    # Actor number
    try:
        actor_id = int(actor_folder.split("_")[1])
    except (IndexError, ValueError):
        continue

    # Odd -> Male
    # Even -> Female
    gender = "male" if actor_id % 2 else "female"

    print(f"\nProcessing {actor_folder} ({gender})")

    for audio_file in sorted(os.listdir(actor_path)):

        if not audio_file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, audio_file)

        try:

            # --------------------------
            # Parse filename
            # --------------------------

            filename = audio_file.replace(".wav", "")
            parts = filename.split("-")

            emotion = emotion_map[parts[2]]

            # --------------------------
            # Extract features
            # --------------------------

            features = extractor.extract_features(file_path)

            # --------------------------
            # Gender Dataset
            # --------------------------

            gender_sample = {}

            for i, value in enumerate(features):
                gender_sample[f"feature_{i}"] = float(value)

            gender_sample["gender"] = gender
            gender_sample["actor"] = actor_id
            gender_sample["file"] = audio_file

            gender_rows.append(gender_sample)

            # --------------------------
            # Emotion Dataset
            # --------------------------

            if gender == "female":

                emotion_sample = {}

                for i, value in enumerate(features):
                    emotion_sample[f"feature_{i}"] = float(value)

                emotion_sample["emotion"] = emotion
                emotion_sample["actor"] = actor_id
                emotion_sample["file"] = audio_file

                emotion_rows.append(emotion_sample)

        except Exception as e:

            print(f"Error processing {audio_file}")
            print(e)

# =====================================================
# CREATE DATAFRAMES
# =====================================================

gender_df = pd.DataFrame(gender_rows)
emotion_df = pd.DataFrame(emotion_rows)

# =====================================================
# SAVE CSV FILES
# =====================================================

gender_output = os.path.join(
    OUTPUT_PATH,
    "gender_features.csv"
)

emotion_output = os.path.join(
    OUTPUT_PATH,
    "emotion_features.csv"
)

gender_df.to_csv(
    gender_output,
    index=False
)

emotion_df.to_csv(
    emotion_output,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Gender samples : {len(gender_df)}")
print(f"Emotion samples: {len(emotion_df)}")

print(f"\nSaved:")
print(gender_output)
print(emotion_output)