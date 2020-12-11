import yaml
import sys

# from lpr.jet_inference import detect as jet_detect
from lpr.tvm_inference import detect as tvm_detect
from lpr.mxnet_inference import detect as mxnet_detect


def parse_settings():
    """ Return dictionary based on settings.yaml file """

    with open("settings.yaml", "r") as data:
        settings = yaml.safe_load(data)
    return settings


if __name__ == "__main__":

    settings = parse_settings()
    if settings["mode"] == "tvm":
        tvm_detect(
            settings["target"],
            settings["language"],
            settings["model_file_dir"],
            str(settings["video_camera"]),
            settings["OpenALPR"],
        )
    elif settings["mode"] == "mxnet":
        mxnet_detect(
            settings["target"],
            settings["language"],
            str(settings["video_camera"]),
            settings["OpenALPR"],
        )
    elif settings["mode"] == "jet_inference":
        jet_detect(
            settings["language"], str(settings["video_camera"]), settings["OpenALPR"]
        )
    else:
        print("Mode not valid")
        sys.exit(1)