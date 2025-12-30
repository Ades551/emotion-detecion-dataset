# Cell 3: Batch HTML Report Generation (One MLP at a Time, with new HTML structure)

from IPython.display import HTML, display
import matplotlib.pyplot as plt
import base64 # For embedding images and audio in HTML
import io
import torch # For softmax
from .empathic_inside_voice_dependencies import *
from .empathic_inside_voice_model import *

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True # Assume it was set in Cell 1 or 2 if used
except ImportError:
    PYDUB_AVAILABLE = False
    # logging.warning("pydub not available for audio embedding in HTML.") # Logging setup in Cell 1/2

from pprint import pp
# --- Helper functions (find_audio_files_in_folder, get_prediction_with_single_mlp,
# --- get_whisper_embedding_for_audio, convert_audio_to_base64_mp3_for_html,
# --- generate_waveform_plot_base64) are assumed to be the same as the previous good version.
# --- Ensure they are defined or copy them here if running this cell standalone after Cell 2.

# Re-define if necessary from previous correct version (or ensure Cell 2 definitions are accessible)
def find_audio_files_in_folder(input_dir: Path) -> List[Path]:
    audio_files = []
    # Assuming SUPPORTED_AUDIO_EXTENSIONS is defined in Cell 2
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(list(input_dir.rglob(f'*{ext}')))
    if not audio_files:
        logging.warning(f"No audio files found in {input_dir.resolve()}.")
    else:
        logging.info(f"Found {len(audio_files)} audio files in {input_dir.resolve()} for HTML report.")
    return sorted(audio_files)

@torch.no_grad()
def get_prediction_with_single_mlp(
    whisper_embedding: torch.Tensor,
    mlp_model: nn.Module,
    current_mlp_device: torch.device
) -> float:
    embedding_for_mlp = whisper_embedding.to(current_mlp_device)
    try:
        current_mlp_dtype = next(mlp_model.parameters()).dtype
        prediction_tensor = mlp_model(embedding_for_mlp.to(current_mlp_dtype))
        return prediction_tensor.item()
    except Exception as e:
        # logging.error(f"Error predicting with a single MLP: {e}", exc_info=True) # logging from Cell 1/2
        return float('nan')

@torch.no_grad()
def get_whisper_embedding_for_audio(
    audio_waveform: np.ndarray,
    loaded_whisper_model: nn.Module,
    loaded_whisper_processor: any,
    current_whisper_device: torch.device
) -> Optional[torch.Tensor]:
    try:
        # Assuming SAMPLING_RATE, WHISPER_SEQ_LEN, WHISPER_EMBED_DIM defined in Cell 2
        input_features = loaded_whisper_processor(
            audio_waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(current_whisper_device).to(loaded_whisper_model.dtype)

        encoder_outputs = loaded_whisper_model.get_encoder()(input_features=input_features)
        embedding = encoder_outputs.last_hidden_state

        current_seq_len = embedding.shape[1]
        if current_seq_len < WHISPER_SEQ_LEN:
            padding = torch.zeros((1, WHISPER_SEQ_LEN - current_seq_len, WHISPER_EMBED_DIM),
                                  device=current_whisper_device, dtype=embedding.dtype)
            embedding = torch.cat((embedding, padding), dim=1)
        elif current_seq_len > WHISPER_SEQ_LEN:
            embedding = embedding[:, :WHISPER_SEQ_LEN, :]
        return embedding
    except Exception as e:
        # logging.error(f"Error generating Whisper embedding: {e}", exc_info=True)
        return None

def convert_audio_to_base64_mp3_for_html(audio_path_str: str) -> Optional[str]:
    if not PYDUB_AVAILABLE: return None
    try:
        # Assuming SAMPLING_RATE defined in Cell 2
        audio = AudioSegment.from_file(audio_path_str)
        audio = audio.set_channels(1).set_frame_rate(SAMPLING_RATE)
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="96k")
        return base64.b64encode(mp3_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        # logging.warning(f"Pydub/ffmpeg error for {audio_path_str}: {e}. No audio player.")
        return None

def generate_waveform_plot_base64(waveform_data: np.ndarray, sr: int) -> str:
    try:
        plt.figure(figsize=(10, 2.5))
        librosa.display.waveshow(waveform_data, sr=sr, color='royalblue', alpha=0.7)
        plt.title("Waveform", fontsize=10)
        plt.xlabel("Time (s)", fontsize=8); plt.ylabel("Amplitude", fontsize=8)
        plt.xticks(fontsize=7); plt.yticks(fontsize=7)
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight'); plt.close()
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.read()).decode('utf-8')
    except Exception as e:
        # logging.warning(f"Waveform plot failed: {e}")
        return ""
# --- End of re-definitions/imports ---


def generate_batch_html_report_updated(
    all_files_scores: Dict[Path, Dict[str, float]], # {filepath: {dim_key: raw_score, ...}}
    target_emotion_keys_list: List[str], # The 40 primary emotion keys
    all_attribute_keys_list: List[str], # All other keys considered attributes
    output_html_path: Path
):
    # HTML Explanation Section (as provided by user)
    explanation_html = """
        <div class="explanation-section">
            <h3>Interpretation of Scores</h3>
            <p>The models predict raw scores. For the 40 Emotional Categories, these raw scores are also used to calculate a normalized <strong>Softmax Probability</strong>, indicating the relative likelihood of each emotion.
            Higher raw scores (shown in parentheses for emotions, or directly for attributes) generally suggest a stronger presence or intensity, aligning with the original annotation scales used during training.</p>

            <h4>Emotional Categories (40)</h4>
            <p><em>Original Annotation Scale: 0 (Not present at all) to 4 (Extremely present).</em><br>
            The table below displays: <strong>Softmax Probability</strong> (Raw Model Score)</p>

            <h4>Attribute Dimensions</h4>
            <p><em>Original Annotation Scale: Varies per dimension (detailed below).</em><br>
            The table for these dimensions displays: Raw Model Score</p>

            <div class="dimension-details-section">
                <h4>Details for Attribute Dimensions:</h4>
                <p><strong>Valence:</strong> <em>Range: -3 (Ext. Negative) to +3 (Ext. Positive).</em> 0=Neutral.</p>
                <p><strong>Arousal:</strong> <em>Range: 0 (Very Calm) to 4 (Very Excited).</em> 2=Neutral.</p>
                <p><strong>Submissive vs. Dominant:</strong> <em>Range: -3 (Ext. Submissive) to +3 (Ext. Dominant).</em> 0=Neutral.</p>
                <p><strong>Age:</strong> <em>Range: 0 (Infant/Toddler) to 6 (Very Old).</em> (e.g., 2=Teenager, 4=Adult).</p>
                <p><strong>Gender:</strong> <em>Range: -2 (Very Masculine) to +2 (Very Feminine).</em> 0=Neutral/Unsure.</p>
                <p><strong>Serious vs. Humorous:</strong> <em>Range: 0 (Very Serious) to 4 (Very Humorous).</em> 2=Neutral.</p>
                <p><strong>Vulnerable vs. Emotionally Detached:</strong> <em>Range: 0 (Very Vulnerable) to 4 (Very Detached).</em> 2=Neutral.</p>
                <p><strong>Confident vs. Hesitant:</strong> <em>Range: 0 (Very Confident) to 4 (Very Hesitant).</em> 2=Neutral.</p>
                <p><strong>Warm vs. Cold:</strong> <em>Range: -2 (Very Cold) to +2 (Very Warm).</em> 0=Neutral.</p>
                <p><strong>Monotone vs. Expressive:</strong> <em>Range: 0 (Very Monotone) to 4 (Very Expressive).</em> 2=Neutral.</p>
                <p><strong>High-Pitched vs. Low-Pitched:</strong> <em>Range: 0 (Very High-Pitched) to 4 (Very Low-Pitched).</em> 2=Neutral.</p>
                <p><strong>Soft vs. Harsh:</strong> <em>Range: -2 (Very Harsh) to +2 (Very Soft).</em> 0=Neutral.</p>
                <p><strong>Authenticity:</strong> <em>Range: 0 (Very Artificial) to 4 (Very Genuine).</em> 2=Neutral.</p>
                <p><strong>Recording Quality:</strong> <em>Range: 0 (Very Low) to 4 (Very High).</em> 2=Decent.</p>
                <p><strong>Background Noise:</strong> <em>Range: 0 (No Noise) to 3 (Intense Noise).</em></p>
            </div>
        </div>
    """

    html_content = [f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Audio Inference Report</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 15px; background-color: #f0f2f5; color: #333; font-size: 14px; }}
    .report-container {{ max-width: 1000px; margin: auto; }}
    .audio-item {{ background-color: #fff; border: 1px solid #e0e0e0; margin-bottom: 20px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.07); }}
    h1 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 10px; }} /* Reduced margin-bottom */
    h2 {{ margin-top: 0; color: #34495e; font-size: 1.3em; border-bottom: 1px dashed #bdc3c7; padding-bottom: 6px; margin-bottom: 12px; }}
    h3 {{ color: #555; font-size: 1.1em; margin-top:15px; margin-bottom:8px;}}
    h4 {{ color: #666; font-size: 1.0em; margin-top:10px; margin-bottom:5px;}}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 0.9em;}}
    th, td {{ border: 1px solid #ddd; padding: 7px; text-align: left; }}
    th {{ background-color: #ecf0f1; font-weight: 600; }}
    .top1 {{ background-color: #ffdddd !important; }} .top2 {{ background-color: #ffe8cc !important; }} .top3 {{ background-color: #ddffdd !important; }}
    audio {{ width: 100%; margin-top: 8px; margin-bottom: 8px; }}
    .waveform-img {{ width:100%; max-width:500px; margin-bottom:10px; border:1px solid #eee; border-radius:4px; display:block; margin-left:auto; margin-right:auto;}}
    .explanation-section {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 25px; border: 1px solid #ced4da;}}
    .explanation-section p {{font-size: 0.95em; line-height: 1.5;}}
    .dimension-details-section p {{ margin-bottom: 3px; font-size:0.9em; }}
    .dimension-details-section strong {{ color: #34495e; }}
</style></head><body><div class="report-container"><h1>Audio Inference Report</h1>
{explanation_html}
"""] # Insert explanation here

    if not all_files_scores:
        html_content.append("<p>No audio files were processed or no results to display.</p>")

    for audio_file_path, raw_predictions in all_files_scores.items():
        audio_file_name = audio_file_path.name
        html_content.append(f"<div class='audio-item'><h2>File: {audio_file_name}</h2>")

        try:
            wf_data, wf_sr = librosa.load(str(audio_file_path), sr=SAMPLING_RATE, mono=True)
            waveform_b64 = generate_waveform_plot_base64(wf_data, wf_sr)
            if waveform_b64:
                html_content.append(f"<img src='data:image/png;base64,{waveform_b64}' alt='Waveform' class='waveform-img'/>")
            del wf_data
        except Exception: pass

        base64_audio_mp3 = convert_audio_to_base64_mp3_for_html(str(audio_file_path))
        if base64_audio_mp3:
            html_content.append(f"<audio controls src='data:audio/mp3;base64,{base64_audio_mp3}'></audio>")
        else:
            html_content.append("<p><i>Audio player not available.</i></p>")

        # Separate raw scores and calculate softmax for emotions
        emotion_raw_scores_dict = {k: raw_predictions.get(k, float('nan')) for k in target_emotion_keys_list}
        attribute_raw_scores_dict = {k: raw_predictions.get(k, float('nan')) for k in all_attribute_keys_list if k in raw_predictions}

        # Softmax calculation
        emotion_scores_for_softmax = [emotion_raw_scores_dict.get(k, -float('inf')) for k in target_emotion_keys_list] # Use -inf for missing, to avoid NaN in softmax
        with torch.no_grad():
            softmax_probs_tensor = torch.softmax(torch.tensor(emotion_scores_for_softmax, dtype=torch.float32), dim=0)
        softmax_probs_dict = {k: softmax_probs_tensor[i].item() for i, k in enumerate(target_emotion_keys_list)}

        # Sort by raw score for highlighting
        sorted_emotions_by_raw_score = sorted(emotion_raw_scores_dict.items(), key=lambda item: item[1], reverse=True)
        top_raw_score_keys = [item[0] for item in sorted_emotions_by_raw_score[:3]]


        html_content.append("<h3>Emotional Categories (40)</h3><table>")
        # Display in the original predefined order
        emotions_in_table_order = target_emotion_keys_list[:]
        num_rows_target = (len(emotions_in_table_order) + 1) // 2
        for i in range(num_rows_target):
            html_content.append("<tr>")
            for col in range(2):
                idx = i + col * num_rows_target
                if idx < len(emotions_in_table_order):
                    key = emotions_in_table_order[idx]
                    softmax_val = softmax_probs_dict.get(key, float('nan'))
                    raw_val = emotion_raw_scores_dict.get(key, float('nan'))

                    css_class = ""
                    if key == (top_raw_score_keys[0] if len(top_raw_score_keys)>0 else None): css_class = "top1"
                    elif key == (top_raw_score_keys[1] if len(top_raw_score_keys)>1 else None): css_class = "top2"
                    elif key == (top_raw_score_keys[2] if len(top_raw_score_keys)>2 else None): css_class = "top3"

                    html_content.append(f"<td class='{css_class}'>{key}</td><td class='{css_class}'>{softmax_val:.4f} ({raw_val:.4f})</td>")
                else:
                    html_content.append("<td></td><td></td>")
            html_content.append("</tr>")
        html_content.append("</table>")

        if attribute_raw_scores_dict:
            html_content.append("<h3>Attribute Dimensions</h3><table>")
            sorted_additional = sorted(attribute_raw_scores_dict.items())
            num_rows_additional = (len(sorted_additional) + 1) // 2
            for i in range(num_rows_additional):
                html_content.append("<tr>")
                for col in range(2):
                    idx = i + col * num_rows_additional
                    if idx < len(sorted_additional):
                        key, score = sorted_additional[idx]
                        html_content.append(f"<td>{key}</td><td>{score:.4f}</td>")
                    else:
                        html_content.append("<td></td><td></td>")
                html_content.append("</tr>")
            html_content.append("</table>")
        html_content.append("</div>")

    html_content.append("</div></body></html>")
    final_html = "".join(html_content)

    try:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        logging.info(f"Batch HTML report saved to: {output_html_path.resolve()}")
    except Exception as e:
        logging.error(f"Error writing HTML report: {e}", exc_info=True)

    return final_html

def predict(mlps_to_use = None, generate_html_file = True, audio_files_for_html_report = []):
    if audio_files_for_html_report == []:
        # --- Main execution for Cell 3 ---
        logging.info("--- Starting Cell 3: Batch HTML Report Generation (Updated) ---")
        audio_files_for_html_report = find_audio_files_in_folder(BATCH_INPUT_AUDIO_FOLDER) # BATCH_INPUT_AUDIO_FOLDER from Cell 2
    aggregated_results_for_html: Dict[Path, Dict[str, float]] = {fp: {} for fp in audio_files_for_html_report}

    if not audio_files_for_html_report:
        logging.warning("No audio files in input folder for HTML report. Skipping.")
        display(HTML("<p><b>No audio files found in input folder. HTML report generation skipped.</b></p>"))
    else:
        if not whisper_model_global or not whisper_processor_global: # from Cell 2
            raise RuntimeError("Whisper model not available for Cell 3.")
        if not all_mlp_model_paths_dict: # from Cell 2
            raise RuntimeError("MLP model paths not available for Cell 3.")

        total_mlps_to_process = len(all_mlp_model_paths_dict)
        mlp_processed_count = 0

        def check_model_usage(model_name):
            if model_name in mlps_to_use:
                return True
            else:
                return False

        for mlp_target_key, mlp_model_path in all_mlp_model_paths_dict.items(): # all_mlp_model_paths_dict from Cell 2
            
            if mlps_to_use is not None:
                if not check_model_usage(mlp_target_key):
                    continue

            mlp_processed_count += 1
            logging.info(f"Processing MLP {mlp_processed_count}/{total_mlps_to_process}: '{mlp_target_key}' for HTML report")

            # MLP_DEVICE, USE_HALF_PRECISION_FOR_MLPS, USE_TORCH_COMPILE_FOR_MLPS from Cell 2
            current_mlp_model = load_single_mlp_model( # load_single_mlp_model from Cell 2
                mlp_model_path, mlp_target_key, MLP_DEVICE,
                USE_HALF_PRECISION_FOR_MLPS, USE_TORCH_COMPILE_FOR_MLPS
            )
            if not current_mlp_model:
                logging.error(f"Skipping MLP '{mlp_target_key}' due to loading error.")
                for audio_f_path in audio_files_for_html_report:
                    aggregated_results_for_html[audio_f_path][mlp_target_key] = float('nan')
                continue

            for audio_file_path in audio_files_for_html_report:
                try:
                    # SAMPLING_RATE, MAX_AUDIO_SECONDS from Cell 2
                    waveform, sr = librosa.load(str(audio_file_path), sr=SAMPLING_RATE, mono=True)
                    max_samples = int(MAX_AUDIO_SECONDS * SAMPLING_RATE)
                    if len(waveform) > max_samples: waveform = waveform[:max_samples]

                    # WHISPER_DEVICE from Cell 2
                    whisper_embedding = get_whisper_embedding_for_audio(
                        waveform, whisper_model_global, whisper_processor_global, WHISPER_DEVICE
                    )
                    del waveform; gc.collect()

                    if whisper_embedding is not None:
                        prediction = get_prediction_with_single_mlp(
                            whisper_embedding, current_mlp_model, MLP_DEVICE
                        )

                        aggregated_results_for_html[audio_file_path][mlp_target_key] = prediction
                        del whisper_embedding;
                    else:
                        aggregated_results_for_html[audio_file_path][mlp_target_key] = float('nan')
                except Exception as e_audio:
                    logging.error(f"Error processing audio {audio_file_path.name} for MLP {mlp_target_key}: {e_audio}")
                    aggregated_results_for_html[audio_file_path][mlp_target_key] = float('nan')

                if WHISPER_DEVICE.type == 'cuda': torch.cuda.empty_cache()

            del current_mlp_model
            gc.collect()
            if MLP_DEVICE.type == 'cuda': torch.cuda.empty_cache()

        # Derive attribute keys (all keys from MLP models that are NOT in the 40 emotion list)
        # all_mlp_model_paths_dict.keys() gives all predictable dimension keys
        # TARGET_EMOTION_KEYS_FOR_REPORT is defined in Cell 2
        all_predictable_keys = set(all_mlp_model_paths_dict.keys())
        emotion_keys_set = set(TARGET_EMOTION_KEYS_FOR_REPORT)
        attribute_keys_derived = sorted(list(all_predictable_keys - emotion_keys_set))

        if generate_html_file:
            # BATCH_OUTPUT_HTML_REPORT_FILE from Cell 2
            
            final_html_output = generate_batch_html_report_updated(
                aggregated_results_for_html,
                TARGET_EMOTION_KEYS_FOR_REPORT, # from Cell 2
                attribute_keys_derived,
                BATCH_OUTPUT_HTML_REPORT_FILE
            )
            display(HTML(final_html_output))
            logging.info("--- Cell 3: Batch HTML Report Generation (Updated) Finished ---")
            
        return aggregated_results_for_html


if __name__ == "__main__":
    predict(generate_html_file=True)
