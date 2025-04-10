import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import soundfile as sf
import hashlib
import os
from datetime import datetime

# ------------------------------------------------------------------
# Hard-Coded Parameters (from grid search and trial & error)
# ------------------------------------------------------------------
TONE_DURATION = 0.11        # seconds
GAP_DURATION = 0.02         # seconds
BASE_FREQ = 500             # Hz
FREQ_RANGE = 1000           # Hz
CHAOS_MOD_RANGE = 349.39    # Hz
NUM_CHAOTIC_SAMPLES = 704
BURN_IN = 900

# Chaotic system parameters from grid search:
DT = 0.005251616433272467   # seconds
A_PARAM = 0.12477067210511437
B_PARAM = 0.2852679643352883
C_PARAM = 6.801715623942842

# ------------------------------------------------------------------
# Section 1: Data Preprocessing
# ------------------------------------------------------------------
def text_to_binary(text, encoding="utf-8"):
    """Convert text to its binary representation."""
    return " ".join(format(byte, "08b") for byte in text.encode(encoding))

def binary_to_text(binary_str, encoding="utf-8"):
    """Convert a binary string (space-separated 8-bit chunks) back to text."""
    byte_list = binary_str.split()
    return bytearray(int(b, 2) for b in byte_list).decode(encoding)

def pad_binary_str(binary_str):
    """Remove spaces and pad the binary string to a multiple of 8 bits."""
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, "0")
    return binary_clean

# ------------------------------------------------------------------
# Helper Function: Downsample Data
# ------------------------------------------------------------------
def downsample_data(data, max_points=1000):
    """Downsample a 1D numpy array 'data' to at most 'max_points'."""
    data = np.array(data)
    n_points = len(data)
    if n_points <= max_points:
        return data
    factor = int(np.ceil(n_points / max_points))
    return data[::factor]

# ------------------------------------------------------------------
# Section 2: Audio Waveform Generation
# ------------------------------------------------------------------
def grouped_binary_to_waveform(binary_str, sample_rate=44100, tone_duration=TONE_DURATION,
                               gap_duration=GAP_DURATION, base_freq=BASE_FREQ, freq_range=FREQ_RANGE):
    """Map each 8-bit group from the binary string to a tone and add gaps."""
    binary_clean = pad_binary_str(binary_str)
    bytes_list = [binary_clean[i: i+8] for i in range(0, len(binary_clean), 8)]
    waveform_segments = []
    for byte_str in bytes_list:
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t_tone)
        gap = np.zeros(int(sample_rate * gap_duration), dtype=np.float32)
        waveform_segments.append(tone)
        waveform_segments.append(gap)
    waveform = np.concatenate(waveform_segments)
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------------------------------------------------------
# Section 3: Chaotic System (Rossler Attractor) Functions
# ------------------------------------------------------------------
def rossler_derivatives(state, a, b, c):
    """Compute the derivatives of the Rossler attractor."""
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)])

def rk4_step(state, dt, a, b, c):
    """Perform one Runge-Kutta 4th order integration step."""
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt/2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt/2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_chaotic_sequence(n, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                              x0=0.1, y0=0.0, z0=0.0, burn_in=BURN_IN):
    """
    Generate a normalized chaotic sequence (x coordinate) from the Rossler attractor.
    """
    state = np.array([x0, y0, z0], dtype=float)
    # Burn-in phase
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    sequence = []
    for _ in range(n):
        state = rk4_step(state, dt, a, b, c)
        sequence.append(state[0])
    sequence = np.array(sequence)
    normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return normalized.tolist()

# ------------------------------------------------------------------
# Section 4: Audio Waveform with Chaotic Modulation (Encryption)
# ------------------------------------------------------------------
def grouped_binary_to_waveform_chaotic(binary_str, sample_rate=44100, tone_duration=TONE_DURATION,
                                       gap_duration=GAP_DURATION, base_freq=BASE_FREQ, freq_range=FREQ_RANGE,
                                       chaos_mod_range=CHAOS_MOD_RANGE, dt=DT, a=A_PARAM, b=B_PARAM,
                                       c=C_PARAM, x0=0.1, y0=0.0, z0=0.0, burn_in=BURN_IN):
    """
    Map each 8-bit group to a tone and modulate its frequency using a chaotic offset.
    """
    binary_clean = pad_binary_str(binary_str)
    bytes_list = [binary_clean[i: i+8] for i in range(0, len(binary_clean), 8)]
    chaotic_sequence = generate_chaotic_sequence(len(bytes_list), dt=dt, a=a, b=b, c=c,
                                                 x0=x0, y0=y0, z0=z0, burn_in=burn_in)
    waveform_segments = []
    for i, byte_str in enumerate(bytes_list):
        byte_val = int(byte_str, 2)
        freq = BASE_FREQ + (byte_val / 255) * FREQ_RANGE
        chaotic_offset = chaotic_sequence[i] * CHAOS_MOD_RANGE
        modulated_freq = freq + chaotic_offset
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * modulated_freq * t_tone)
        gap = np.zeros(int(sample_rate * GAP_DURATION), dtype=np.float32)
        waveform_segments.append(tone)
        waveform_segments.append(gap)
    waveform = np.concatenate(waveform_segments)
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------------------------------------------------------
# Section 5: Audio Synthesis & Storage
# ------------------------------------------------------------------
def synthesize_and_store_audio(waveform, sample_rate=44100, filename_prefix="oscilLOCK_audio", file_format="WAV"):
    """Save the waveform to an audio file using a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{file_format.lower()}"
    storage_dir = "audio_storage"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    full_path = os.path.join(storage_dir, filename)
    sf.write(full_path, waveform, sample_rate, format=file_format)
    return full_path

def convert_waveform_to_audio_bytes(waveform, sample_rate, file_format="WAV"):
    """Convert the waveform into audio bytes for playback/download."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format=file_format)
    return buf.getvalue()

# ------------------------------------------------------------------
# Section 6: Visualization Functions (Plotly)
# ------------------------------------------------------------------
def create_waveform_figure(waveform, sample_rate, title="Waveform", zoom_range=None, max_points=1000):
    """Create a downsampled Plotly line chart of the waveform."""
    time_vector = np.linspace(0, len(waveform)/sample_rate, len(waveform), endpoint=False)
    time_ds = downsample_data(time_vector, max_points)
    waveform_ds = downsample_data(waveform, max_points)
    fig = go.Figure(data=go.Scatter(x=time_ds, y=waveform_ds, mode="lines", name="Waveform"))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    if zoom_range:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])
    return fig

def create_fft_figure(waveform_plain, waveform_chaotic, sample_rate, max_points=1000):
    """Create a Plotly figure comparing the FFTs of the plain and chaotic waveforms."""
    N = len(waveform_plain)
    fft_plain = np.abs(np.fft.rfft(waveform_plain))
    fft_chaotic = np.abs(np.fft.rfft(waveform_chaotic))
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    freqs_ds = downsample_data(freqs, max_points)
    fft_plain_ds = downsample_data(fft_plain, max_points)
    fft_chaotic_ds = downsample_data(fft_chaotic, max_points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs_ds, y=fft_plain_ds, mode="lines", name="Encoded Audio FFT"))
    fig.add_trace(go.Scatter(x=freqs_ds, y=fft_chaotic_ds, mode="lines", name="Encrypted Audio FFT"))
    fig.update_layout(title="FFT Comparison", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    fig.update_xaxes(range=[0, 1500])
    return fig

def plot_correlation_coefficient(waveform_plain, waveform_chaotic, max_points=1000):
    """Create a scatter plot of encoded vs. encrypted audio amplitudes."""
    corr = np.corrcoef(waveform_plain, waveform_chaotic)[0,1]
    wp_ds = downsample_data(waveform_plain, max_points)
    we_ds = downsample_data(waveform_chaotic, max_points)
    fig = go.Figure(data=go.Scatter(x=wp_ds, y=we_ds, mode="markers", marker=dict(size=4)))
    fig.update_layout(title=f"Scatter Plot (Correlation: {corr:.2f})", 
                      xaxis_title="Encoded Audio Amplitude", yaxis_title="Encrypted Audio Amplitude")
    return fig

def create_chaotic_phase_plot_3d(binary_str, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                                 x0=0.1, y0=0.0, z0=0.0, max_points=1000, burn_in=BURN_IN):
    """Generate a 3D phase plot (x,y,z) from the chaotic trajectory after burn-in."""
    binary_clean = binary_str.replace(" ", "")
    n_bytes = len(binary_clean) // 8
    state = np.array([x0, y0, z0], dtype=float)
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    trajectory = []
    for _ in range(n_bytes):
        state = rk4_step(state, dt, a, b, c)
        trajectory.append(state.copy())
    trajectory = np.array(trajectory)
    trajectory_ds = trajectory[::max(1, int(len(trajectory)/max_points))]
    fig = go.Figure(data=[go.Scatter3d(
        x=trajectory_ds[:,0],
        y=trajectory_ds[:,1],
        z=trajectory_ds[:,2],
        mode="markers",
        marker=dict(size=3, color=trajectory_ds[:,0], colorscale="Viridis")
    )])
    fig.update_layout(title="3D Chaotic Phase Plot", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig

def create_chaotic_phase_plots_2d(binary_str, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                                  x0=0.1, y0=0.0, z0=0.0, max_points=1000, burn_in=BURN_IN):
    """Generate 2D phase plots (XY, YZ, ZX) from the chaotic trajectory."""
    binary_clean = binary_str.replace(" ", "")
    n_bytes = len(binary_clean) // 8
    state = np.array([x0, y0, z0], dtype=float)
    for _ in range(burn_in):
        state = rk4_step(state, dt, a, b, c)
    trajectory = []
    for _ in range(n_bytes):
        state = rk4_step(state, dt, a, b, c)
        trajectory.append(state.copy())
    trajectory = np.array(trajectory)
    trajectory_ds = trajectory[::max(1, int(len(trajectory)/max_points))]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["XY Phase Plot", "YZ Phase Plot", "ZX Phase Plot"])
    fig.add_trace(go.Scatter(x=trajectory_ds[:,0], y=trajectory_ds[:,1], mode="markers", marker=dict(color="blue", size=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=trajectory_ds[:,1], y=trajectory_ds[:,2], mode="markers", marker=dict(color="green", size=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=trajectory_ds[:,2], y=trajectory_ds[:,0], mode="markers", marker=dict(color="red", size=3)), row=1, col=3)
    fig.update_layout(title="2D Chaotic Phase Plots", showlegend=False)
    return fig

def plot_chaotic_sequence(chaotic_sequence, max_points=1000):
    """Plot the raw chaotic sequence using downsampling."""
    seq_ds = downsample_data(chaotic_sequence, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(seq_ds))), y=seq_ds, mode="lines", name="Chaotic Sequence"))
    fig.update_layout(title="Raw Chaotic Sequence", xaxis_title="Index", yaxis_title="Value")
    return fig

def plot_quantized_chaotic_sequence(chaotic_sequence, max_points=1000):
    """Plot the quantized chaotic sequence (8-bit values) using downsampling."""
    chaotic_array = np.array(chaotic_sequence)
    quantized = np.uint8(255 * chaotic_array)
    quantized_ds = downsample_data(quantized, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(quantized_ds))), y=quantized_ds,
                                    mode="lines+markers", name="Quantized Chaotic Sequence"))
    fig.update_layout(title="Quantized Chaotic Sequence", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

def plot_audio_features(audio_features, max_points=1000):
    """Plot quantized audio feature values using downsampling."""
    audio_features_ds = downsample_data(audio_features, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(audio_features_ds))), y=audio_features_ds,
                                    mode="lines+markers", name="Audio Features"))
    fig.update_layout(title="Audio Features (Quantized)", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

# ------------------------------------------------------------------
# Section 7: Key Generation Functions
# ------------------------------------------------------------------
def derive_initial_conditions(passphrase):
    """Derive initial conditions from the SHA‑256 hash of the passphrase."""
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()
    norm_const = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(hash_digest[0:21], 16) / norm_const
    y0 = int(hash_digest[21:42], 16) / norm_const
    z0 = int(hash_digest[42:64], 16) / norm_const
    return x0, y0, z0

def get_audio_feature_values(waveform, num_samples=NUM_CHAOTIC_SAMPLES):
    """Extract amplitude samples from the waveform and quantize them to 8-bit."""
    indices = np.linspace(0, len(waveform)-1, num_samples, dtype=int)
    samples = waveform[indices]
    quantized = np.uint8(255 * ((samples - samples.min()) / (samples.max() - samples.min() + 1e-6)))
    return quantized

def sample_audio_features(waveform, num_samples=NUM_CHAOTIC_SAMPLES):
    """Extract and quantize audio features; return as a byte string."""
    quantized = get_audio_feature_values(waveform, num_samples)
    return quantized.tobytes()

def generate_chaotic_key(passphrase, waveform, chaotic_params, num_chaotic_samples=NUM_CHAOTIC_SAMPLES):
    """
    Generate a cryptographic key by combining:
      1. A chaotic sequence derived from the passphrase.
      2. Audio features extracted from the encoded waveform.
    Returns the key (SHA‑256 hash) and the raw chaotic sequence.
    """
    x0, y0, z0 = derive_initial_conditions(passphrase)
    dt, a, b, c = chaotic_params
    chaotic_sequence = generate_chaotic_sequence(num_chaotic_samples, dt=dt, a=a, b=b, c=c,
                                                 x0=x0, y0=y0, z0=z0, burn_in=BURN_IN)
    chaotic_array = np.array(chaotic_sequence)
    chaotic_quantized = np.uint8(255 * chaotic_array)
    chaotic_bytes = chaotic_quantized.tobytes()
    audio_bytes = sample_audio_features(waveform, num_chaotic_samples)
    combined = chaotic_bytes + audio_bytes
    key = hashlib.sha256(combined).hexdigest()
    return key, chaotic_sequence

# ------------------------------------------------------------------
# Section 8: Streamlit UI with Hard-Coded Parameters and Sidebar Form
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="oscilLOCK", layout="wide")
    st.title("oscilLOCK - Multimodal Audio Based Encryptor")
    
    # Sidebar: Input form with an "Enter" button
    with st.sidebar:
        st.title("Enter Message")
        with st.form(key="message_form"):
            user_text = st.text_area("Enter text to encrypt:", "Hello, oscilLOCK!")
            passphrase = st.text_input("Enter passphrase for key generation:", "DefaultPassphrase", type="password")
            submit_button = st.form_submit_button("Enter")
    
    if submit_button and user_text:
        sample_rate = 44100
        
        # Data Preprocessing
        binary_output = text_to_binary(user_text)
        recovered_text = binary_to_text(binary_output)
        
        # Generate the encoded (plain) waveform
        waveform_encoded, _ = grouped_binary_to_waveform(
            binary_output,
            sample_rate=sample_rate,
            tone_duration=TONE_DURATION,
            gap_duration=GAP_DURATION,
            base_freq=BASE_FREQ,
            freq_range=FREQ_RANGE
        )
        
        # Derive initial conditions from passphrase
        derived_x0, derived_y0, derived_z0 = derive_initial_conditions(passphrase)
        
        # Generate the encrypted (chaotic-modulated) waveform
        waveform_encrypted, _ = grouped_binary_to_waveform_chaotic(
            binary_output,
            sample_rate=sample_rate,
            tone_duration=TONE_DURATION,
            gap_duration=GAP_DURATION,
            base_freq=BASE_FREQ,
            freq_range=FREQ_RANGE,
            chaos_mod_range=CHAOS_MOD_RANGE,
            dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
            x0=derived_x0, y0=derived_y0, z0=derived_z0,
            burn_in=BURN_IN
        )
        
        # Key Generation
        chaotic_params = (DT, A_PARAM, B_PARAM, C_PARAM)
        derived_key, chaotic_sequence = generate_chaotic_key(
            passphrase, waveform_encoded, chaotic_params, num_chaotic_samples=NUM_CHAOTIC_SAMPLES
        )
        audio_features = get_audio_feature_values(waveform_encoded)
        
        # Layout with Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Preprocessing & Encoding", 
            "Encryption Module", 
            "Comparison", 
            "Key Generation", 
            "Storage"
        ])
        
        with tab1:
            st.header("Preprocessing & Encoding")
            st.markdown("**Binary Representation:**")
            st.code(binary_output)
            st.markdown(f"**Recovered Text:** {recovered_text}")
            st.markdown("**Encoded Audio:**")
            audio_encoded = convert_waveform_to_audio_bytes(waveform_encoded, sample_rate, file_format="WAV")
            st.audio(audio_encoded, format="audio/wav", start_time=0)
        
        with tab2:
            st.header("Encryption Module")
            st.markdown("**Encrypted Audio:**")
            encrypted_audio_bytes = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate, file_format="WAV")
            st.audio(encrypted_audio_bytes, format="audio/wav", start_time=0)
        
        with tab3:
            st.header("Comparison")
            st.subheader("FFT Comparison")
            fig_fft = create_fft_figure(waveform_encoded, waveform_encrypted, sample_rate)
            st.plotly_chart(fig_fft, use_container_width=True)
            st.subheader("Scatter Plot (Correlation)")
            fig_corr = plot_correlation_coefficient(waveform_encoded, waveform_encrypted)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.subheader("3D Chaotic Phase Plot")
            fig_phase_3d = create_chaotic_phase_plot_3d(
                binary_output, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                x0=derived_x0, y0=derived_y0, z0=derived_z0, burn_in=BURN_IN
            )
            st.plotly_chart(fig_phase_3d, use_container_width=True)
            st.subheader("2D Chaotic Phase Plots (XY, YZ, ZX)")
            fig_phase_2d = create_chaotic_phase_plots_2d(
                binary_output, dt=DT, a=A_PARAM, b=B_PARAM, c=C_PARAM,
                x0=derived_x0, y0=derived_y0, z0=derived_z0, burn_in=BURN_IN
            )
            st.plotly_chart(fig_phase_2d, use_container_width=True)
        
        with tab4:
            st.header("Key Generation")
            st.markdown("**Derived Cryptographic Key:**")
            st.code(derived_key)
            st.markdown("### Visualizing the Key Generation Process")
            fig_chaotic_raw = plot_chaotic_sequence(chaotic_sequence)
            st.plotly_chart(fig_chaotic_raw, use_container_width=True)
            fig_chaotic_quant = plot_quantized_chaotic_sequence(chaotic_sequence)
            st.plotly_chart(fig_chaotic_quant, use_container_width=True)
            fig_audio_features = plot_audio_features(audio_features)
            st.plotly_chart(fig_audio_features, use_container_width=True)
            st.markdown("The key is generated by concatenating the quantized chaotic sequence with audio features extracted from the encoded waveform, then hashing the result with SHA‑256.")
        
        with tab5:
            st.header("Storage")
            st.markdown("Download the encrypted audio file:")
            file_format = "WAV"
            mime_type = "audio/wav"
            download_audio = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate, file_format=file_format)
            st.download_button(label=f"Download Encrypted Audio ({file_format})",
                               data=download_audio,
                               file_name=f"encrypted_audio.{file_format.lower()}",
                               mime=mime_type)

if __name__ == "__main__":
    main()
