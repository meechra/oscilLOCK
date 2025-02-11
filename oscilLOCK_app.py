import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
import soundfile as sf

# ------------------ Module 1: Data Preprocessing ------------------
def text_to_binary(text, encoding='utf-8'):
    """Convert text to its binary representation."""
    byte_array = text.encode(encoding)
    binary_str = ' '.join(format(byte, '08b') for byte in byte_array)
    return binary_str

def binary_to_text(binary_str, encoding='utf-8'):
    """Convert a binary string back to text."""
    byte_list = binary_str.split()
    byte_array = bytearray(int(b, 2) for b in byte_list)
    text = byte_array.decode(encoding)
    return text

# ------------------ Module 2: Grouped Binary-to-Waveform (Plain Mapping) ------------------
def grouped_binary_to_waveform_plain(binary_str, sample_rate=44100, tone_duration=0.2, gap_duration=0.05,
                                     base_freq=300, freq_range=700):
    """
    Convert the binary string (grouped into 8-bit chunks) into an audio waveform 
    without chaotic modulation.
    """
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, '0')
    bytes_list = [binary_clean[i:i+8] for i in range(0, len(binary_clean), 8)]
    
    waveform = np.array([], dtype=np.float32)
    for byte_str in bytes_list:
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t_tone)
        gap_samples = int(sample_rate * gap_duration)
        gap = np.zeros(gap_samples, dtype=np.float32)
        waveform = np.concatenate((waveform, tone, gap))
        
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------ Module 3: Chaotic Function Integration using Rossler Attractor ------------------
def generate_chaotic_sequence_rossler(n, dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Generate a sequence of chaotic values using the Rossler attractor.
    
    The system is integrated via Euler’s method. The x-component is sampled and normalized to [0, 1].
    """
    sequence = []
    x, y, z = x0, y0, z0
    for _ in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        sequence.append(x)
    sequence = np.array(sequence)
    sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return sequence.tolist()

# ------------------ Module 4: Grouped Binary-to-Waveform with Chaotic Modulation (Encryption) ------------------
def grouped_binary_to_waveform_chaotic(binary_str, sample_rate=44100, tone_duration=0.2, gap_duration=0.05,
                                       base_freq=300, freq_range=700, chaos_mod_range=100,
                                       dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Convert the binary string into an audio waveform with chaotic modulation.
    For each byte, the tone frequency is shifted by a chaotic offset from the Rossler attractor.
    """
    binary_clean = binary_str.replace(" ", "")
    if len(binary_clean) % 8 != 0:
        binary_clean = binary_clean.ljust(((len(binary_clean) // 8) + 1) * 8, '0')
    bytes_list = [binary_clean[i:i+8] for i in range(0, len(binary_clean), 8)]
    
    chaos_seq = generate_chaotic_sequence_rossler(len(bytes_list), dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
    
    waveform = np.array([], dtype=np.float32)
    for i, byte_str in enumerate(bytes_list):
        byte_val = int(byte_str, 2)
        freq = base_freq + (byte_val / 255) * freq_range
        chaotic_offset = chaos_seq[i] * chaos_mod_range
        modulated_freq = freq + chaotic_offset
        t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
        tone = np.sin(2 * np.pi * modulated_freq * t_tone)
        gap_samples = int(sample_rate * gap_duration)
        gap = np.zeros(gap_samples, dtype=np.float32)
        waveform = np.concatenate((waveform, tone, gap))
    
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
    return waveform, time_vector

# ------------------ Visualization Functions ------------------
def create_waveform_figure(waveform, sample_rate, title="Waveform", zoom_range=None):
    """Return a Plotly figure of the waveform."""
    time_vector = np.linspace(0, len(waveform)/sample_rate, len(waveform), endpoint=False)
    fig = go.Figure(data=go.Scatter(x=time_vector, y=waveform, mode='lines', name='Waveform'))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude", hovermode='x')
    if zoom_range:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])
    return fig

def create_fft_figure(waveform_plain, waveform_chaotic, sample_rate):
    """Return a Plotly figure comparing the FFTs of the plain and chaotic waveforms."""
    N = len(waveform_plain)
    fft_plain = np.abs(np.fft.rfft(waveform_plain))
    fft_chaotic = np.abs(np.fft.rfft(waveform_chaotic))
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=fft_plain, mode='lines', name='Encoded Audio FFT'))
    fig.add_trace(go.Scatter(x=freqs, y=fft_chaotic, mode='lines', name='Encrypted Audio FFT'))
    fig.update_layout(title="FFT Comparison", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    fig.update_xaxes(range=[0, 1500])
    return fig

def create_difference_figure(waveform_plain, waveform_chaotic, sample_rate, zoom_range=(0, 0.005)):
    """Return a Plotly figure of the difference waveform (encrypted - encoded) zoomed in."""
    diff_waveform = waveform_chaotic - waveform_plain
    fig = create_waveform_figure(diff_waveform, sample_rate, title="Difference Waveform (Zoomed-In)", zoom_range=zoom_range)
    return fig

def convert_waveform_to_audio_bytes(waveform, sample_rate):
    """Convert a numpy waveform to WAV bytes for audio playback."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format='WAV')
    return buf.getvalue()

# ------------------ Streamlit Interface ------------------
def main():
    # Set page configuration
    st.set_page_config(page_title="oscilLOCK", layout="wide")
    
    # Sidebar for input and parameters
    st.sidebar.header("Input & Parameters")
    user_text = st.sidebar.text_input("Enter text to encrypt:", "Hello, oscilLOCK!")
    
    st.sidebar.markdown("### Audio Parameters")
    tone_duration = st.sidebar.slider("Tone Duration (sec)", 0.1, 0.5, 0.2)
    gap_duration = st.sidebar.slider("Gap Duration (sec)", 0.01, 0.1, 0.05)
    base_freq = st.sidebar.number_input("Base Frequency (Hz)", 100, 1000, 300)
    freq_range = st.sidebar.number_input("Frequency Range (Hz)", 100, 2000, 700)
    chaos_mod_range = st.sidebar.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
    
    st.title("oscilLOCK: Rhythm-Based Encryption Pipeline")
    st.markdown("""
    **oscilLOCK** transforms your input text through a modular pipeline:
    
    1. **Data Preprocessing:** Converts text to a binary string.
    2. **Encoding Module:** Maps the binary data to an audio waveform (Encoded Output).
    3. **Encryption Module:** Applies chaotic modulation (via a Rossler attractor) to produce an encrypted audio waveform.
    4. **Cross Verification:** Compares the outputs to highlight the effects of encryption.
    """)
    
    if user_text:
        # Data Preprocessing
        binary_output = text_to_binary(user_text)
        recovered_text = binary_to_text(binary_output)
        
        # Create tabs for each pipeline module
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preprocessing", "Encoding Module", "Encryption Module", "Cross Verification"])
        
        with tab1:
            st.header("Data Preprocessing")
            st.markdown("The input text is converted into a binary representation:")
            st.subheader("Binary Representation")
            st.code(binary_output)
            st.markdown(f"**Recovered Text:** {recovered_text}")
        
        sample_rate = 44100  # Fixed audio sample rate
        
        # Encoding Module: Generate encoded (plain) waveform and audio
        waveform_encoded, _ = grouped_binary_to_waveform_plain(
            binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
            base_freq=base_freq, freq_range=freq_range
        )
        
        with tab2:
            st.header("Encoding Module")
            st.markdown("This module maps the binary data to an audio waveform (Encoded Output).")
            col1, col2 = st.columns(2)
            zoom_range = (0, 0.005)  # Zoom in on first 5 ms
            with col1:
                st.subheader("Encoded Waveform (Zoomed-In)")
                fig_encoded = create_waveform_figure(waveform_encoded, sample_rate, title="Encoded Waveform", zoom_range=zoom_range)
                st.plotly_chart(fig_encoded, use_container_width=True)
            with col2:
                st.subheader("Encoded Audio")
                audio_encoded = convert_waveform_to_audio_bytes(waveform_encoded, sample_rate)
                st.audio(audio_encoded, format='audio/wav', start_time=0)
        
        # Encryption Module: Generate encrypted (chaotic) waveform and audio
        waveform_encrypted, _ = grouped_binary_to_waveform_chaotic(
            binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
            base_freq=base_freq, freq_range=freq_range, chaos_mod_range=chaos_mod_range,
            dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0
        )
        
        with tab3:
            st.header("Encryption Module")
            st.markdown("The encoded waveform is modulated with chaotic offsets (via the Rossler attractor) to produce the Encrypted Output.")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Encrypted Waveform (Zoomed-In)")
                fig_encrypted = create_waveform_figure(waveform_encrypted, sample_rate, title="Encrypted Waveform", zoom_range=zoom_range)
                st.plotly_chart(fig_encrypted, use_container_width=True)
            with col2:
                st.subheader("Encrypted Audio")
                audio_encrypted = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate)
                st.audio(audio_encrypted, format='audio/wav', start_time=0)
        
        # Cross Verification: Compare outputs
        with tab4:
            st.header("Cross Verification")
            st.markdown("The difference waveform and FFT comparison highlight the changes introduced by the encryption process.")
            st.subheader("Difference Waveform (Encrypted - Encoded)")
            fig_diff = create_difference_figure(waveform_encoded, waveform_encrypted, sample_rate, zoom_range=zoom_range)
            st.plotly_chart(fig_diff, use_container_width=True)
            st.subheader("FFT Comparison")
            fig_fft = create_fft_figure(waveform_encoded, waveform_encrypted, sample_rate)
            st.plotly_chart(fig_fft, use_container_width=True)

if __name__ == "__main__":
    main()
