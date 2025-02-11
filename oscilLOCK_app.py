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
    
    Integrated via Euler’s method; the x-component is normalized to [0, 1].
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
    Each byte’s tone frequency is shifted by a chaotic offset (via the Rossler attractor).
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
    time_vector = np.linspace(0, len(waveform) / sample_rate, len(waveform), endpoint=False)
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

def create_chaotic_phase_plot(binary_str, dt=0.01, a=0.2, b=0.2, c=5.7, x0=0.1, y0=0.0, z0=0.0):
    """
    Generate and return a phase plot (scatter plot of x[i+1] vs x[i]) of the chaotic sequence
    produced by the Rossler attractor. The number of points equals the number of 8-bit chunks in the binary string.
    """
    binary_clean = binary_str.replace(" ", "")
    n_bytes = len(binary_clean) // 8
    chaotic_sequence = generate_chaotic_sequence_rossler(n_bytes, dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
    x_vals = chaotic_sequence[:-1]
    y_vals = chaotic_sequence[1:]
    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=dict(color='red', size=8)))
    fig.update_layout(title="Chaotic Phase Plot", xaxis_title="x[i]", yaxis_title="x[i+1]")
    return fig

def convert_waveform_to_audio_bytes(waveform, sample_rate):
    """Convert a numpy waveform to WAV bytes for audio playback."""
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format='WAV')
    return buf.getvalue()

# ------------------ Streamlit Interface ------------------
def main():
    st.set_page_config(page_title="oscilLOCK", layout="wide")
    
    # Sidebar form for input and customization controls
    with st.sidebar.form(key="input_form"):
        st.sidebar.header("Input & Customization")
        user_text = st.text_input("Enter text to encrypt:", "Hello, oscilLOCK!", max_chars=500)
        
        st.sidebar.markdown("### Audio Parameters")
        tone_duration = st.sidebar.slider("Tone Duration (sec)", 0.1, 0.5, 0.2)
        gap_duration = st.sidebar.slider("Gap Duration (sec)", 0.01, 0.1, 0.05)
        base_freq = st.sidebar.number_input("Base Frequency (Hz)", 100, 1000, 300)
        freq_range = st.sidebar.number_input("Frequency Range (Hz)", 100, 2000, 700)
        chaos_mod_range = st.sidebar.number_input("Chaos Mod Range (Hz)", 0, 500, 100)
        
        with st.sidebar.expander("Advanced Chaotic Parameters"):
            dt = st.slider("dt", 0.001, 0.05, 0.01, step=0.001)
            a = st.slider("a", 0.1, 1.0, 0.2, step=0.1)
            b = st.slider("b", 0.1, 1.0, 0.2, step=0.1)
            c = st.slider("c", 1.0, 10.0, 5.7, step=0.1)
            x0 = st.number_input("x0", value=0.1, step=0.1)
            y0 = st.number_input("y0", value=0.0, step=0.1)
            z0 = st.number_input("z0", value=0.0, step=0.1)
        
        submit_button = st.form_submit_button(label="Enter")
    
    if submit_button and user_text:
        with st.spinner("Processing..."):
            # Data Preprocessing
            binary_output = text_to_binary(user_text)
            recovered_text = binary_to_text(binary_output)
            sample_rate = 44100  # Fixed audio sample rate
            
            # Generate waveforms
            waveform_encoded, _ = grouped_binary_to_waveform_plain(
                binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
                base_freq=base_freq, freq_range=freq_range
            )
            waveform_encrypted, _ = grouped_binary_to_waveform_chaotic(
                binary_output, sample_rate=sample_rate, tone_duration=tone_duration, gap_duration=gap_duration,
                base_freq=base_freq, freq_range=freq_range, chaos_mod_range=chaos_mod_range,
                dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0
            )
        
        # Create tabs for the modular pipeline
        tab1, tab2, tab3 = st.tabs(["Preprocessing & Encoding", "Encryption Module", "Comparison"])
        
        with tab1:
            st.header("Preprocessing & Encoding")
            st.markdown("**Binary Representation:**")
            st.code(binary_output)
            st.markdown(f"**Recovered Text:** {recovered_text}")
            st.markdown("**Encoded Audio:**")
            audio_encoded = convert_waveform_to_audio_bytes(waveform_encoded, sample_rate)
            st.audio(audio_encoded, format='audio/wav', start_time=0)
        
        with tab2:
            st.header("Encryption Module")
            st.markdown("**Encrypted Audio:**")
            audio_encrypted = convert_waveform_to_audio_bytes(waveform_encrypted, sample_rate)
            st.audio(audio_encrypted, format='audio/wav', start_time=0)
        
        with tab3:
            st.header("Comparison")
            zoom_range = (0, 0.005)  # Zoom in on first 5 ms for time-domain plots
            st.subheader("Difference Waveform (Encrypted - Encoded)")
            fig_diff = create_difference_figure(waveform_encoded, waveform_encrypted, sample_rate, zoom_range=zoom_range)
            st.plotly_chart(fig_diff, use_container_width=True)
            
            st.subheader("FFT Comparison")
            fig_fft = create_fft_figure(waveform_encoded, waveform_encrypted, sample_rate)
            st.plotly_chart(fig_fft, use_container_width=True)
            
            st.subheader("Chaotic Dynamics Visualization")
            fig_phase = create_chaotic_phase_plot(binary_output, dt=dt, a=a, b=b, c=c, x0=x0, y0=y0, z0=z0)
            st.plotly_chart(fig_phase, use_container_width=True)

if __name__ == "__main__":
    main()
