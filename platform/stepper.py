# Continuous one-direction spin.
#
# Wiring (BCM numbering):
#   IN1 -> GPIO17 (pin 11)
#   IN2 -> GPIO18 (pin 12)
#   IN3 -> GPIO27 (pin 13)
#   IN4 -> GPIO22 (pin 15)

# Prereqs:
#   sudo apt install python3-lgpio
#
# Run:
#   python3 stepper.py

import time
import lgpio

IN1, IN2, IN3, IN4 = 17, 18, 27, 22
PINS = (IN1, IN2, IN3, IN4)

STEP_DELAY_S = 0.0013  # 1.3 ms per step

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


def main():
    chip = lgpio.gpiochip_open(0)
    for pin in PINS:
        lgpio.gpio_claim_output(chip, pin, 0)

    try:
        next_t = time.perf_counter()
        while True:
            for pattern in STEPS:
                for pin, value in zip(PINS, pattern):
                    lgpio.gpio_write(chip, pin, value)
                next_t += STEP_DELAY_S
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        for pin in PINS:
            lgpio.gpio_write(chip, pin, 0)
            lgpio.gpio_free(chip, pin)
        lgpio.gpiochip_close(chip)


if __name__ == "__main__":
    main()
