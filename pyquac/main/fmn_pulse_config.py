# config functions for pulsed spectroscopy
import textwrap


def z_control_and_readout(amplitude, channel, length, fade_length, readout_seq):
    return textwrap.dedent(
        f"""
    const z_pulse_length_part = {fade_length};
    const z_pulse_length = {length};
    const amp = {amplitude};
    const channel = {channel};
    
    wave w_gauss = amp*gauss(z_pulse_length_part,z_pulse_length_part/2,z_pulse_length_part/8);
    
    wave w_rise = cut(w_gauss,0,z_pulse_length_part/2-1);
    wave w_fall = cut(w_gauss,z_pulse_length_part/2,z_pulse_length_part-1);
    wave w_flat = rect(z_pulse_length,amp);
    wave w_rect = join(w_rise,w_flat,w_fall);
    
    playWave(channel,w_rect);
    waitWave();

    // Play wave 2
    {readout_seq}
    """
    )


def xy_control(amplitude, channel_I, channel_Q, length, z_length):
    return textwrap.dedent(
        f"""
    const xy_pulse_length = {length};
    const amp = {amplitude};
    const channel_I = {channel_I}
    const channel_Q = {channel_Q}
    const length_wait = {int(z_length / 2)}

    wave null = zeros(length_wait);
    wave w_xy_gauss = amp*gauss(xy_pulse_length,xy_pulse_length/2,xy_pulse_length/8);
    wave w_xy_gauss_await = join(null, w_xy_gauss);

    playWave(channel_I, w_xy_gauss, channel_Q, w_xy_gauss);
    """
    )


def readout_control(amplitude, channel_I, channel_Q, fade_length, length):
    return textwrap.dedent(
        f"""
    const fade_length = {fade_length}; //fade pulse length
    const readout_pulse_length = {length}; //equivalent to 625 ns
    const amp = {amplitude};
    const channel_I = {channel_I}
    const channel_Q = {channel_Q}
    
    wave w_gauss = amp*gauss(fade_length,fade_length/2,fade_length/8);

    wave w_rise = cut(w_gauss,0,fade_length/2-1);
    wave w_fall = cut(w_gauss,fade_length/2,fade_length-1);
    wave w_flat = amp*rect(readout_pulse_length,1);
    wave w_rect = join(w_rise,w_flat,w_fall); //flat accurate readout pulse
    
    playWave(channel_I, w_rect, channel_Q, w_rect); // readout pulse playback
    """
    )


def z_readout_combined(
    amplitude_z,
    channel_z,
    length_z,
    fade_length_z,
    amplitude_r,
    channel_i_r,
    channel_q_r,
    length_r,
    fade_length_r,
    averages_count,
    wait_time,
):
    return textwrap.dedent(
        f"""
    // Z-control pulse config
    const fade_length_z = {fade_length_z};
    const length_z = {length_z};
    const amplitude_z = {amplitude_z};
    const channel_z = {channel_z};

    wave w_gauss_z = amplitude_z*gauss(fade_length_z,fade_length_z/2,fade_length_z/8);
    wave w_rise_z = cut(w_gauss_z,0,fade_length_z/2-1);
    wave w_fall_z = cut(w_gauss_z,fade_length_z/2,fade_length_z-1);
    wave w_flat_z = rect(length_z,amplitude_z);
    wave w_rect_z = join(w_rise_z,w_flat_z,w_fall_z);

    // Readout pulse config
    const fade_length_r = {fade_length_r}; //fade pulse length
    const length_r = {length_r}; //equivalent to 625 ns
    const amplitude_r = {amplitude_r};
    const channel_i_r = {channel_i_r}
    const channel_q_r = {channel_q_r}

    wave w_gauss_r = amplitude_r*gauss(fade_length_r,fade_length_r/2,fade_length_r/8);

    wave w_rise_r = cut(w_gauss_r,0,fade_length_r/2-1);
    wave w_fall_r = cut(w_gauss_r,fade_length_r/2,fade_length_r-1);
    wave w_flat_r = amplitude_r*rect(length_r,1);
    wave w_rect_r = join(w_rise_r, w_flat_r, w_fall_r); //flat accurate readout pulse

    // Playback config
    const averages_count = {averages_count};
    const wait_time = {wait_time}

    repeat(averages_count) {{
    wait(wait_time);

    playWave(channel_z,w_rect_z); // Z pulse playback
    waitWave();

    setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER); // trigger only for modulation
    setTrigger(AWG_MONITOR_TRIGGER); // trigger after gauss, reading only readout pulse

    playWave(channel_i_r, w_rect_r, channel_q_r, w_rect_r); // readout pulse playback
    }}
    """
    )
