import os


def get_attributes(file_name: str):
    file_details = file_name.split("_")
    attributes = {k: v for k, v in zip(file_details[6::2], file_details[7::2])}
    attributes["section"] = file_details[1]
    attributes["domain"] = file_details[2]
    attributes["label"] = file_details[4]
    return attributes


def get_groupings(audio_dir: str):
    file_list = os.listdir(audio_dir)
    domains = set()
    sections = set()
    for file_name in file_list:
        section = file_name.split("_")[1]
        domain = file_name.split("_")[2]
        sections.add(section)
        domains.add(domain)
    return sections, domains
