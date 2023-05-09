import os


def read_weebit(path: str):
    data = []

    # Get texts for 7-8 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel2"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel2\\" + file_name)
        lines = f.readlines()
        lines.pop(len(lines) - 1)
        text = ""

        for line in lines:
            text += line

        data.append({"text": text, "label": "7-8"})

    # Get texts for 8-9 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel3\\" + file_name)
        lines = f.readlines()
        lines.pop(len(lines) - 1)
        text = ""

        for line in lines:
            text += line

        data.append({"text": text, "label": "8-9"})

    # Get texts for 9-10 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel4"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel4\\" + file_name)
        lines = f.readlines()
        lines.pop(len(lines) - 1)
        text = ""

        for line in lines:
            text += line

        data.append({"text": text, "label": "9-10"})

    # Get texts for 11-14 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\BitKS3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\BitKS3\\" + file_name)
        lines = f.readlines()
        text = ""

        if lines[0][0] == '>':
            lines.pop(0)

        if "The BBC is not responsible for the content of external internet sites.\n" in lines:
            lines.remove("The BBC is not responsible for the content of external internet sites.\n")
        if "This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n" in lines:
            lines.remove("This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n")

        for line in lines:
            text += line

        data.append({"text": text, "label": "11-14"})

    # Get texts for 15-16 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\BitGCSE"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\BitGCSE\\" + file_name)
        lines = f.readlines()
        text = ""

        if "The BBC is not responsible for the content of external internet sites.\n" in lines:
            lines.remove("The BBC is not responsible for the content of external internet sites.\n")
        if "This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n" in lines:
            lines.remove("This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n")
        while "You have disabled Javascript, or are not running Javascript on this browser. Go to the\n" in lines:
            lines.remove("You have disabled Javascript, or are not running Javascript on this browser. Go to the\n")

        for line in lines:
            text += line

        data.append({"text": text, "label": "15-16"})

    return data


def read_newsela(path: str):
    pass


def split_train_test(data):
    pass
