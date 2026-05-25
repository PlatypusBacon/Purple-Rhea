# Wiring
#   IN1 -> GPIO17 (pin 11)
#   IN2 -> GPIO18 (pin 12)
#   IN3 -> GPIO27 (pin 13)
#   IN4 -> GPIO22 (pin 15)

import time
import lgpio

IN1, IN2, IN3, IN4 = 17, 18, 27, 22
PINS = (IN1, IN2, IN3, IN4)

STEP_DELAY_S       = 0.0013   # 1.3 ms per half-step
STEPPER_STEPS_PER_REV = 4096  # 28BYJ-48 half-step output

DRIVER_TEETH = 15
DRIVEN_TEETH = 115           

INCREMENT_DEG      = 10.0     # how far to move the output gear
TOTAL_DEG          = 360.0    
PAUSE_BETWEEN_S    = 2.0      # total time per increment
HOLD_TORQUE        = False    

STEPS = (
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
)


def deenergize(chip):
    for pin in PINS:
        lgpio.gpio_write(chip, pin, 0)


def move_steps(chip, n, phase):
    """Advance n half-steps from the given phase index. Returns the next phase index."""
    next_t = time.perf_counter()
    for _ in range(n):
        pattern = STEPS[phase]
        for pin, value in zip(PINS, pattern):
            lgpio.gpio_write(chip, pin, value)
        next_t += STEP_DELAY_S
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()
        phase = (phase + 1) % 8
    return phase


def main():
    gear_ratio = DRIVEN_TEETH / DRIVER_TEETH                       # output rev = stepper rev / ratio
    steps_per_output_rev = STEPPER_STEPS_PER_REV * gear_ratio       # ~31403 for 15:115
    steps_per_increment_f = steps_per_output_rev * INCREMENT_DEG / 360.0

    print(f"Gear ratio (driven/driver): {gear_ratio:.4f}")
    print(f"Half-steps per output revolution: {steps_per_output_rev:.2f}")
    print(f"Half-steps per {INCREMENT_DEG}° increment: {steps_per_increment_f:.2f}")

    chip = lgpio.gpiochip_open(0)
    for pin in PINS:
        lgpio.gpio_claim_output(chip, pin, 0)

    num_increments = None if TOTAL_DEG is None else round(TOTAL_DEG / INCREMENT_DEG)
    if num_increments is not None:
        print(f"Total increments to perform: {num_increments}")

    phase = 0
    accumulated = 0.0  # fractional-step accumulator to prevent drift across many increments
    done = 0
    try:
        while num_increments is None or done < num_increments:
            cycle_start = time.perf_counter()
            accumulated += steps_per_increment_f
            n = int(accumulated)
            accumulated -= n
            phase = move_steps(chip, n, phase)
            done += 1
            if num_increments is not None and done >= num_increments:
                break
            if not HOLD_TORQUE:
                deenergize(chip)
            remaining = PAUSE_BETWEEN_S - (time.perf_counter() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        deenergize(chip)
        for pin in PINS:
            lgpio.gpio_free(chip, pin)
        lgpio.gpiochip_close(chip)


if __name__ == "__main__":
    main()
