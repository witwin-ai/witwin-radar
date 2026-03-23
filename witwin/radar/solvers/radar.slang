[AutoPyBindCUDA]
[CUDAKernel]
void frameMIMO_kernel(
    TensorView<double> t_sample,                    // [T]  time sampling = 256 
    TensorView<float> total_lengths,                // [F, TX, RX, N]
    TensorView<float> amplitudes,                   // [F, TX, RX, N]
    float fc, float slope, float phi, int max_N,    // max_N is the maximum number of points in these chirps
    TensorView<double> out_real,                    // [TX, RX, F, T]
    TensorView<double> out_imag                     // [TX, RX, F, T]
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint pair_id = cudaBlockIdx().y;
    uint num_rx = (uint)total_lengths.size(2);
    uint tx_id = pair_id / num_rx;
    uint rx_id = pair_id % num_rx;
    uint chirp_id = cudaBlockIdx().z;

    if (t_idx >= t_sample.size(0)) return;

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (uint i = 0; i < max_N; ++i) {
        float amp = amplitudes[chirp_id, tx_id, rx_id, i];
        if (amp == 0.0) continue;

        float distance = total_lengths[chirp_id, tx_id, rx_id, i];
        float tof = distance / c0;  // time of flight

        double time = t_val - (double)tof;
        double rx_phase = pi2 * ((double)fc * time + 0.5 * (double)slope * time * time) + (double)phi;
        double rx_real = cos(rx_phase);
        double rx_imag = sin(rx_phase);  // received signal

        double tx_phase = pi2 * ((double)fc * t_val + 0.5 * (double)slope * t_val * t_val) + (double)phi;
        double tx_real = cos(tx_phase);
        double tx_imag = sin(tx_phase);  // transmitted signal

        double mixed_real = tx_real * rx_real + tx_imag * rx_imag;
        double mixed_imag = tx_imag * rx_real - tx_real * rx_imag;  // mixed signal

        sum_real += mixed_real * (double)amp; 
        sum_imag += mixed_imag * (double)amp; // sum of all reflections point at this sample point 
    }

    out_real[tx_id, rx_id, chirp_id, t_idx] = sum_real;
    out_imag[tx_id, rx_id, chirp_id, t_idx] = sum_imag;
}


// Chirp kernel - internal sum version (memory efficient, less parallel)
// Each thread handles one time sample and loops through ALL targets
// Output: [T] - final result directly
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_internal(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    TensorView<double> out_real,    // [T] output real
    TensorView<double> out_imag     // [T] output imag
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;

    int T = t_sample.size(0);
    if (t_idx >= T) return;

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];

    double tx_phase = pi2 * ((double)fc * t_val + 0.5 * (double)slope * t_val * t_val);
    double tx_real = cos(tx_phase);
    double tx_imag = sin(tx_phase);

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int i = 0; i < num_targets; ++i) {
        float distance = distances[i];
        float amp = amplitudes[i];

        if (amp == 0.0) continue;

        float tof = distance / c0;
        double time = t_val - (double)tof;
        double rx_phase = pi2 * ((double)fc * time + 0.5 * (double)slope * time * time);
        double rx_real = cos(rx_phase);
        double rx_imag = sin(rx_phase);

        double mixed_real = tx_real * rx_real + tx_imag * rx_imag;
        double mixed_imag = tx_imag * rx_real - tx_real * rx_imag;

        sum_real += mixed_real * (double)amp;
        sum_imag += mixed_imag * (double)amp;
    }

    out_real[t_idx] = sum_real;
    out_imag[t_idx] = sum_imag;
}

// Chirp kernel - external sum version (chunked, more parallel, more memory)
// Grid: (num_t_blocks, num_target_chunks, 1)
// Each block processes a chunk of targets and writes to partial output
// Output: [num_target_chunks, T] - partial sums to be reduced in Python
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_chunked(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    int targets_per_chunk,
    TensorView<double> out_real,    // [num_chunks, T] output real
    TensorView<double> out_imag     // [num_chunks, T] output imag
)
{
    uint t_idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    uint chunk_idx = cudaBlockIdx().y;

    int T = t_sample.size(0);
    if (t_idx >= T) return;

    // Target range for this chunk
    int target_start = chunk_idx * targets_per_chunk;
    int target_end = min(target_start + targets_per_chunk, num_targets);

    float c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double t_val = t_sample[t_idx];

    // TX waveform phase
    double tx_phase = pi2 * ((double)fc * t_val + 0.5 * (double)slope * t_val * t_val);
    double tx_real = cos(tx_phase);
    double tx_imag = sin(tx_phase);

    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int i = target_start; i < target_end; ++i) {
        float distance = distances[i];
        float amp = amplitudes[i];

        if (amp == 0.0) continue;

        float tof = distance / c0;

        double time = t_val - (double)tof;
        double rx_phase = pi2 * ((double)fc * time + 0.5 * (double)slope * time * time);
        double rx_real = cos(rx_phase);
        double rx_imag = sin(rx_phase);

        double mixed_real = tx_real * rx_real + tx_imag * rx_imag;
        double mixed_imag = tx_imag * rx_real - tx_real * rx_imag;

        sum_real += mixed_real * (double)amp;
        sum_imag += mixed_imag * (double)amp;
    }

    out_real[chunk_idx, t_idx] = sum_real;
    out_imag[chunk_idx, t_idx] = sum_imag;
}

// Chirp kernel - per-target chunked version
// Each thread handles one target and loops through ALL time samples
// Output: [num_chunks, T] - partial sums to be reduced in Python
[AutoPyBindCUDA]
[CUDAKernel]
void chirp_kernel_per_target(
    TensorView<double> t_sample,    // [T] time samples
    TensorView<float> distances,    // [N] round-trip distances
    TensorView<float> amplitudes,   // [N] amplitudes
    float fc,
    float slope,
    int num_targets,
    int T,
    int targets_per_chunk,
    TensorView<float> out_real,     // [num_chunks, T] output real
    TensorView<float> out_imag      // [num_chunks, T] output imag
)
{
    uint thread_idx = cudaThreadIdx().x;
    uint chunk_idx = cudaBlockIdx().x;

    uint target_idx = chunk_idx * targets_per_chunk + thread_idx;
    if (target_idx >= num_targets) return;

    double c0 = 299792458.0;
    double pi2 = 6.283185307179586;

    double distance = (double)distances[target_idx];
    double amp = (double)amplitudes[target_idx];

    if (amp == 0.0) return;

    double tof = distance / c0;

    // Loop through all time samples
    for (int t_idx = 0; t_idx < T; ++t_idx) {
        double t_val = t_sample[t_idx];

        double tx_phase = pi2 * ((double)fc * t_val + 0.5 * (double)slope * t_val * t_val);
        double tx_real = cos(tx_phase);
        double tx_imag = sin(tx_phase);

        double time = t_val - tof;
        double rx_phase = pi2 * ((double)fc * time + 0.5 * (double)slope * time * time);
        double rx_real = cos(rx_phase);
        double rx_imag = sin(rx_phase);

        double mixed_real = tx_real * rx_real + tx_imag * rx_imag;
        double mixed_imag = tx_imag * rx_real - tx_real * rx_imag;

        float contrib_real = (float)(mixed_real * amp);
        float contrib_imag = (float)(mixed_imag * amp);

        // Atomic add within chunk (threads in same block write to same chunk output)
        float oldVal;
        out_real.InterlockedAdd(uint2(chunk_idx, t_idx), contrib_real, oldVal);
        out_imag.InterlockedAdd(uint2(chunk_idx, t_idx), contrib_imag, oldVal);
    }
}

