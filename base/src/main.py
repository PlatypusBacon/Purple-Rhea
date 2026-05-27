"""
Entry point for the Purple-Rhea base node.
Swap serial_receiver for mqtt_receiver when MQTT is ready.
"""

from comms.disk_session import start_disk_session
from comms.session import start_session
from pipeline import runner
import config


def main(on_frame_captured=None, on_pipeline_progress=None):
    print("Base Node running")
    session = start_session(on_frame_captured=on_frame_captured)
    #session = start_disk_ble_session(on_frame_captured=on_frame_captured)
    #session = start_disk_session()


    if not session.is_complete():
        print(f"\nWARNING: Only {len(session)}/{config.TOTAL_FRAMES} frames received.")
        if on_frame_captured is None:
            ans = input("Run pipeline anyway? [y/N]: ")
            if ans.lower() != "y":
                return
        else:
            print("Running pipeline with incomplete session (web mode).")

    obj_path = runner.run(session, on_progress=on_pipeline_progress)
    print(f"\nDone. Model saved to: {obj_path}")


if __name__ == "__main__":
    main()