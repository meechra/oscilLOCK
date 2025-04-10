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
# Section 1: Data Preprocessing
# ------------------------------------------------------------------
def text_to_binary(text, encoding="utf-8"):
    """
    Convert a text string to its binary representation.
    Each byte is represented as an 8-bit binary string.
    """
    return " ".join(format(byte, "08b") for byte in text.encode(encoding))

def binary_to_text(binary_str, encoding="utf-8"):
    """
    Convert a binary string (space-separated 8-bit chunks) back to text.
    """
    byte_list = binary_str.split()
    return bytearray(int(b, 2) for b in byte_list).decode(encoding)

def pad_binary_str(binary_str):
    """
    Remove spaces from the binary string and pad it to ensure its length is a multiple of 8.
    """
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, "0")
    return binary_clean

# ------------------------------------------------------------------
# Helper Function: Downsample Data
# ------------------------------------------------------------------
def downsample_data(data, max_points=1000):
    """
    Downsample the 1D numpy array 'data' to at most 'max_points' data points.
    """
    data = np.array(data)
    n_points = len(data)
    if n_points <= max_points:
        return data
    factor = int(np.ceil(n_points / max_points))
    return data[::factor]

# ------------------------------------------------------------------
# Section 2: Audio Waveform Generation
# ------------------------------------------------------------------
def grouped_binary_to_waveform(binary_str, sample_rate=44100, tone_duration=0.2,
                               gap_duration=0.05, base_freq=300, freq_range=700):
    """
    Map each 8-bit group from the binary string to a tone whose frequency is computed
    by a linear mapping. A silence gap is added after each tone.
    Returns the waveform and corresponding time vector.
    """
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
    """
    Compute the derivatives of the Rossler attractor.
    """
    x, y, z = state
    return np.array([-y - z, x + a * y, b + z * (x - c)])

def rk4_step(state, dt, a, b, c):
    """
    Perform a single Runge-Kutta 4th order integration step.
    """
    k1 = rossler_derivatives(state, a, b, c)
    k2 = rossler_derivatives(state + dt / 2 * k1, a, b, c)
    k3 = rossler_derivatives(state + dt / 2 * k2, a, b, c)
    k4 = rossler_derivatives(state + dt * k3, a, b, c)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_chaotic_sequence(n, dt=0.01, a=0.2, b=0.2, c=5.7,
                              x0=0.1, y0=0.0, z0=0.0, burn_in=100):
    """
    Generate a normalized chaotic sequence (using the x coordinate) from the Rossler attractor.
    
    Steps:
      1. Execute 'burn_in' RK4 steps to allow transients to settle.
      2. Collect 'n' successive samples (using the x coordinate).
      3. Normalize the sequence to [0, 1].
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
def grouped_binary_to_waveform_chaotic(binary_str, sample_rate=44100, tone_duration=0.2,
                                       gap_duration=0.05, base_freq=300, freq_range=700,
                                       chaos_mod_range=100, dt=0.01, a=0.2, b=0.2,
                                       c=5.7, x0=0.1, y0=0.0, z0=0.0, burn_in=100):
    """
    Map each 8-bit group from the binary string to a tone and modulate its frequency
    with a chaotic offset computed from the Rossler attractor.
    Returns the chaotic-modulated waveform and time vector.
    """
    binary_clean = pad_binary_str(binary_str)
    bytes_list = [binary_clean[i: i+8] for i in range(0, len(binary_clean), 8)]
    chaotic_sequence = generate_chaotic_sequence(len(bytes_list), dt=dt, a=a, b=b, c=c,
                                                 x0=x0, y0=y0, z0=z0, burn_in=burn_in)
    
    waveform_segments = []
    for i, byte_str in enumerate(bytes_list):
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        chaotic_offset = chaotic_sequence[i] * chaos_mod_range
        modulated_freq = freq + chaotic_offset
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * modulated_freq * t_tone)
        gap = np.zeros(int(sample_rate * gap_duration), dtype=np.float32)
        waveform_segments.append(tone)
        waveform_segments.append(gap)
        
    waveform = np.concatenate(waveform_segments)
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------------------------------------------------------
# Section 5: Audio Synthesis & Storage
# ------------------------------------------------------------------
def synthesize_and_store_audio(waveform, sample_rate=44100, filename_prefix="oscilLOCK_audio", file_format="WAV"):
    """
    Write the provided waveform to an audio file using a timestamped filename.
    Returns the full file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.{file_format.lower()}"
    storage_dir = "audio_storage"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    full_path = os.path.join(storage_dir, filename)
    sf.write(full_path, waveform, sample_rate, format=file_format)
    return full_path

def convert_waveform_to_audio_bytes(waveform, sample_rate, file_format="WAV"):
    """
    Convert the waveform into audio bytes for playback or download.
    """
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format=file_format)
    return buf.getvalue()

# ------------------------------------------------------------------
# Section 6: Visualization Functions (Plotly)
# ------------------------------------------------------------------
def create_waveform_figure(waveform, sample_rate, title="Waveform", zoom_range=None, max_points=1000):
    """
    Create a downsampled Plotly line chart of the waveform versus time.
    """
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    time_ds = downsample_data(time_vector, max_points)
    waveform_ds = downsample_data(waveform, max_points)
    
    fig = go.Figure(data=go.Scatter(x=time_ds, y=waveform_ds, mode="lines", name="Waveform"))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude", hovermode="x")
    if zoom_range:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])
    return fig

def create_fft_figure(waveform_plain, waveform_chaotic, sample_rate, max_points=1000):
    """
    Create a Plotly figure comparing the FFTs of the plain and chaotic waveforms.
    Downsample FFT data if necessary.
    """
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

def plot_entropy_comparison(waveform_plain, waveform_chaotic, bins=256):
    """
    Create a bar chart comparing the Shannon entropy of the plain and chaotic waveforms.
    """
    entropy_plain = compute_shannon_entropy(waveform_plain, bins)
    entropy_chaotic = compute_shannon_entropy(waveform_chaotic, bins)
    fig = go.Figure(data=[go.Bar(x=["Encoded Audio", "Encrypted Audio"],
                                 y=[entropy_plain, entropy_chaotic])])
    fig.update_layout(title="Entropy Comparison", xaxis_title="Waveform", yaxis_title="Shannon Entropy (bits)")
    return fig

def compute_shannon_entropy(signal, bins=256):
    """
    Compute the Shannon entropy of the signal.
    """
    histogram, _ = np.histogram(signal, bins=bins, density=True)
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))
    return entropy

def plot_correlation_coefficient(waveform_plain, waveform_chaotic, max_points=1000):
    """
    Create a scatter plot of encoded vs. encrypted audio amplitudes using downsampled data.
    Displays the Pearson correlation coefficient.
    """
    corr = np.corrcoef(waveform_plain, waveform_chaotic)[0, 1]
    wp_ds = downsample_data(waveform_plain, max_points)
    we_ds = downsample_data(waveform_chaotic, max_points)
    fig = go.Figure(data=go.Scatter(x=wp_ds, y=we_ds, mode="markers", marker=dict(size=4), name="Data Points"))
    fig.update_layout(title=f"Scatter Plot (Correlation: {corr:.2f})", 
                      xaxis_title="Encoded Audio Amplitude", yaxis_title="Encrypted Audio Amplitude")
    return fig

def create_chaotic_phase_plot_3d(binary_str, dt=0.01, a=0.2, b=0.2, c=5.7,
                                 x0=0.1, y0=0.0, z0=0.0, max_points=1000, burn_in=100):
    """
    Generate a 3D phase plot (x, y, z) from the chaotic trajectory after burn-in.
    """
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
        marker=dict(size=3, color=trajectory_ds[:,0], colorscale="Viridis"),
    )])
    fig.update_layout(title="3D Chaotic Phase Plot", scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z"
    ))
    return fig

def create_chaotic_phase_plots_2d(binary_str, dt=0.01, a=0.2, b=0.2, c=5.7,
                                  x0=0.1, y0=0.0, z0=0.0, max_points=1000, burn_in=100):
    """
    Generate 2D phase plots for the chaotic trajectory (XY, YZ, ZX), arranged in a row.
    """
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
    fig.add_trace(go.Scatter(x=trajectory_ds[:,0], y=trajectory_ds[:,1], mode="markers",
                             marker=dict(color="blue", size=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=trajectory_ds[:,1], y=trajectory_ds[:,2], mode="markers",
                             marker=dict(color="green", size=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=trajectory_ds[:,2], y=trajectory_ds[:,0], mode="markers",
                             marker=dict(color="red", size=3)), row=1, col=3)
    fig.update_layout(title="2D Chaotic Phase Plots", showlegend=False)
    return fig

def plot_chaotic_sequence(chaotic_sequence, max_points=1000):
    """
    Plot the raw chaotic sequence using downsampling.
    """
    seq_ds = downsample_data(chaotic_sequence, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(seq_ds))), y=seq_ds, mode="lines", name="Chaotic Sequence"))
    fig.update_layout(title="Raw Chaotic Sequence", xaxis_title="Index", yaxis_title="Value")
    return fig

def plot_quantized_chaotic_sequence(chaotic_sequence, max_points=1000):
    """
    Plot the quantized chaotic sequence (8-bit values) using downsampling.
    """
    chaotic_array = np.array(chaotic_sequence)
    quantized = np.uint8(255 * chaotic_array)
    quantized_ds = downsample_data(quantized, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(quantized_ds))), y=quantized_ds,
                                    mode="lines+markers", name="Quantized Chaotic Sequence"))
    fig.update_layout(title="Quantized Chaotic Sequence", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

def plot_audio_features(audio_features, max_points=1000):
    """
    Plot quantized audio feature values using downsampling.
    """
    audio_features_ds = downsample_data(audio_features, max_points)
    fig = go.Figure(data=go.Scatter(x=list(range(len(audio_features_ds))), y=audio_features_ds,
                                    mode="lines+markers", name="Audio Features"))
    fig.update_layout(title="Audio Features (Quantized)", xaxis_title="Index", yaxis_title="Value (0-255)")
    return fig

# ------------------------------------------------------------------
# Section 7: Key Generation Functions
# ------------------------------------------------------------------
def derive_initial_conditions(passphrase):
    """
    Derive initial conditions from the SHA‑256 hash of the passphrase.
    Split into three parts for x0, y0, and z0.
    """
    hash_digest = hashlib.sha256(passphrase.encode()).hexdigest()  # 64 hex characters
    norm_const = float(0xFFFFFFFFFFFFFFFFFFFFF)
    x0 = int(hash_digest[0:21], 16) / norm_const
    y0 = int(hash_digest[21:42], 16) / norm_const
    z0 = int(hash_digest[42:64], 16) / norm_const
    return x0, y0, z0

def get_audio_feature_values(waveform, num_samples=128):
    """
    Extract a number of amplitude samples from the waveform,
    quantize to 8-bit values, and return as a numpy array.
    """
    indices = np.linspace(0, len(waveform)-1, num_samples, dtype=int)
    samples = waveform[indices]
    quantized = np.uint8(255 * ((samples - samples.min()) / (samples.max() - samples.min() + 1e-6)))
    return quantized

def sample_audio_features(waveform, num_samples=128):
    """
    Extract and quantize audio features from the waveform,
    returning them as a byte string.
    """
    quantized = get_audio_feature_values(waveform, num_samples)
    return quantized.tobytes()

def generate_chaotic_key(passphrase, waveform, chaotic_params, num_chaotic_samples=128):
    """
    Generate a cryptographic key by combining:
      1. A chaotic sequence derived from the passphrase.
      2. Audio features from the encoded waveform.
    Concatenate the quantized values and hash with SHA‑256.
    Returns the hexadecimal key and the raw chaotic sequence.
    """
    x0, y0, z0 = derive_initial_conditions(passphrase)
    dt, a, b, c = chaotic_params
    chaotic_sequence = generate_chaotic_sequence(num_chaotic_samples, dt=dt, a=a, b=b, c=c,
                                                 x0=x0, y0=y0, z0=z0, burn_in=100)
    chaotic_array = np.array(chaotic_sequence)
    chaotic_quantized = np.uint8(255 * chaotic_array)
    chaotic_bytes = chaotic_quantized.tobytes()
    audio_bytes = sample_audio_features(waveform, num_chaotic_samples)
    combined = chaotic_bytes + audio_bytes
    key = hashlib.sha256(combined).hexdigest()
    return key, chaotic_sequence

# ------------------------------------------------------------------
# Section 8: Streamlit Interface
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="oscilLOCK", layout="wide")
    
    # Sidebar: Input parameters and controls
    st.sidebar.title("CONTROL PANEL")
    with st.sidebar.form(key="input_form"):
        user_text = st.text_input("Enter text to encrypt:", "Hello, oscilLOCK!", max_chars=500)
        passphrase = st.text_input("Enter passphrase for key generation:", "DefaultPassphrase", type="password")
        
        st.markdown("### Audio Parameters")
        tone_duration = st.slider("Tone Duration (sec)", 0.1, 0.5, 0.2)
        gap_duration = st.slider("Gap Duration (sec)", 0.01, 0.1, 0.05)
        base_freq = st.number_input("Base Frequency (Hz)", 100, 1000, 300)
        freq_range = st.number_input("Frequency Range (Hz)", 100, 2000, 700)
        chaos_mod_range = st.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
        
        st.markdown("### Chaotic Key Generation")
        num_chaotic_samples = st.slider("Number of Chaotic Samples", 64, 1024, 128, step=64)
        
        with st.expander("Advanced Chaotic Parameters"):
            dt = st.slider("dt", 0.001, 0.05, 0.01, step=0.001)
            a = st.slider("a", 0.1, 1.0, 0.2, step=0.1)
            b = st.slider("b", 0.1, 1.0, 0.2, step=0.1)
            c = st.slider("c", 1.0, 10.0, 5.7, step=0.1)
            burn_in = st.slider("Burn-in Steps", 0, 1000, 100, step=50)
            # Initial conditions derived solely from the passphrase.
        submit_button = st.form_submit_button(label="Enter")
    
    if submit_button and user_text:
        with st.spinner("Processing..."):
            # Data Preprocessing
            binary_output = text_to_binary(user_text)
            recovered_text = binary_to_text(binary_output)
            sample_rate = 44100
            
            # Generate plain (encoded) waveform
            waveform_encoded, _ = grouped_binary_to_waveform(
                binary_output,
                sample_rate=sample_rate,
                tone_duration=tone_duration,
                gap_duration=gap_duration,
                base_freq=base_freq,
                freq_range=freq_range
            )
            
            # Derive initial conditions from passphrase
            derived_x0, derived_y0, derived_z0 = derive_initial_conditions(passphrase)
            
            # Generate chaotic (encrypted) waveform
            waveform_encrypted, _ = grouped_binary_to_waveform_chaotic(
                binary_output,
                sample_rate=sample_rate,
                tone_duration=tone_duration,
                gap_duration=gap_duration,
                base_freq=base_freq,
                freq_range=freq_range,
                chaos_mod_range=chaos_mod_range,
                dt=dt, a=a, b=b, c=c,
                x0=derived_x0, y0=derived_y0, z0=derived_z0,
                burn_in=burn_in
            )
            
            # Key Generation
            chaotic_params = (dt, a, b, c)
            derived_key, chaotic_sequence = generate_chaotic_key(
                passphrase, waveform_encoded, chaotic_params, num_chaotic_samples
            )
            audio_features = get_audio_feature_values(waveform_encoded, num_samples=num_chaotic_samples)
            
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
            st.subheader("Entropy Comparison")
            fig_entropy = plot_entropy_comparison(waveform_encoded, waveform_encrypted)
            st.plotly_chart(fig_entropy, use_container_width=True)
            
            st.subheader("Correlation Coefficient Analysis")
            fig_corr = plot_correlation_coefficient(waveform_encoded, waveform_encrypted)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.subheader("FFT Comparison")
            fig_fft = create_fft_figure(waveform_encoded, waveform_encrypted, sample_rate)
            st.plotly_chart(fig_fft, use_container_width=True)
            
            st.subheader("3D Chaotic Phase Visualization")
            fig_phase_3d = create_chaotic_phase_plot_3d(
                binary_output, dt=dt, a=a, b=b, c=c,
                x0=derived_x0, y0=derived_y0, z0=derived_z0,
                burn_in=burn_in
            )
            st.plotly_chart(fig_phase_3d, use_container_width=True)
            
            st.subheader("2D Chaotic Phase Plots (XY, YZ, ZX)")
            fig_phase_2d = create_chaotic_phase_plots_2d(
                binary_output, dt=dt, a=a, b=b, c=c,
                x0=derived_x0, y0=derived_y0, z0=derived_z0,
                burn_in=burn_in
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
            st.markdown("The key is generated by concatenating the quantized chaotic sequence (derived from your passphrase) with audio features extracted from the encoded waveform, then hashing the result with SHA‑256.")
        
        with tab5:
            st.header("Storage")
            st.markdown("Select a format to download the encrypted audio file:")
            file_format = st.selectbox("Download Format", options=["WAV", "FLAC", "OGG"], index=0)
            if file_format.upper() == "WAV":
                mime_type = "audio/wav"
            elif file_format.upper() == "FLAC":
                mime_type = "audio/flac"
            elif file_format.upper() == "OGG":
                mime_type = "audio/ogg"
            else:
                mime_type = "audio/wav"
            download_audio = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate, file_format=file_format.upper())
            st.download_button(label=f"Download Encrypted Audio ({file_format.upper()})",
                               data=download_audio,
                               file_name=f"encrypted_audio.{file_format.lower()}",
                               mime=mime_type)

if __name__ == "__main__":
    main()
