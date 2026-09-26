import numpy as np


def flops(image_size, batch):
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)

    conv_flops = 17_712 * batch * image_size**2
    linear_flops = 313_344 * batch

    return conv_flops + linear_flops

def memory(image_size, batch):
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)

    weights_memory = 4_161_296
    peak_activations = 52 * batch * image_size**2

    return weights_memory + peak_activations

def bytes_moved(image_size, batch):
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)

    activations = 364 * batch * image_size**2
    head_activations = 8_592 * batch
    weights = 4_161_296

    return activations + head_activations + weights

def latency(image_size, batch, theta):
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)

    launch_latency = theta["launch_latency"]
    compute_rate = theta["compute_rate"]
    memory_bandwidth = theta["memory_bandwidth"]

    bs2 = batch * image_size**2

    layers = [
        ("conv1", 2_352 * bs2, 44 * bs2 + 18_816),
        ("relu1", 0, 64 * bs2),
        ("maxpool", 0, 40 * bs2),

        ("conv2", 6_400 * bs2, 24 * bs2 + 204_800),
        ("relu2", 0, 32 * bs2),

        ("conv3", 2_304 * bs2, 24 * bs2 + 294_912),
        ("relu3", 0, 16 * bs2),

        ("conv4", 1_024 * bs2, 24 * bs2 + 131_072),
        ("relu4", 0, 32 * bs2),

        ("conv5", 4_608 * bs2, 20 * bs2 + 2_359_296),
        ("relu5", 0, 8 * bs2),

        ("conv6", 1_024 * bs2, 12 * bs2 + 524_288),
        ("relu6", 0, 16 * bs2),

        (
            "global_avg_pool",
            0,
            8 * bs2 + 2_048 * batch,
        ),
        (
            "linear1",
            262_144 * batch,
            3_072 * batch + 525_312,
        ),
        ("relu7", 0, 2_048 * batch),
        (
            "linear2",
            51_200 * batch,
            1_424 * batch + 102_800,
        ),
    ]

    total_latency = np.zeros_like(bs2, dtype=float)

    for _, layer_flops, layer_bytes in layers:
        compute_time = layer_flops / compute_rate
        memory_time = layer_bytes / memory_bandwidth

        layer_latency = (
            launch_latency
            + np.maximum(compute_time, memory_time)
        )

        total_latency += layer_latency

    return total_latency

def energy(image_size, batch, theta_energy):
    image_size = np.asarray(image_size)
    batch = np.asarray(batch)

    base_power = theta_energy["base_power"]
    energy_per_flop = theta_energy["energy_per_flop"]
    energy_per_byte = theta_energy["energy_per_byte"]
    latency_theta = theta_energy["latency_theta"]

    forward_latency = latency(
        image_size,
        batch,
        latency_theta,
    )

    base_energy = base_power * forward_latency
    compute_energy = energy_per_flop * flops(
        image_size,
        batch,
    )
    memory_energy = energy_per_byte * bytes_moved(
        image_size,
        batch,
    )

    return base_energy + compute_energy + memory_energy