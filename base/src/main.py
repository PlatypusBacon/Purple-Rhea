"""
Entry point for the Purple-Rhea base node.
Swap serial_receiver for mqtt_receiver when MQTT is ready.
"""

from comms.session import start_session
from pipeline import runner
import config


def main():
    print("Base Node running")
    session = start_session()

    session.save_all_jpegs(config.IMAGE_CACHE)   # save for debug

    if not session.is_complete():
        print(f"\nWARNING: Only {len(session)}/{config.TOTAL_FRAMES} frames received.")
        ans = input("Run pipeline anyway? [y/N]: ")
        if ans.lower() != "y":
            return

    obj_path = runner.run(session)
    print(f"\nDone. Model saved to: {obj_path}")


if __name__ == "__main__":
    main()